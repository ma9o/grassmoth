/** TypeScript property readers in knip's production web project, for check_fields.py. */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import ts from "typescript";

type ExportedField = {
  identity: string;
  name: string;
  owner: string;
  aliases: string[];
  components: string[];
};

const root = resolve(Bun.argv[2] ?? resolve(import.meta.dirname, "../../../.."));
const web = resolve(root, "apps/grassmoth-client");
const configFile = ts.readConfigFile(resolve(web, "tsconfig.json"), ts.sys.readFile);
if (configFile.error) throw new Error(ts.flattenDiagnosticMessageText(configFile.error.messageText, "\n"));
const config = ts.parseJsonConfigFileContent(configFile.config, ts.sys, web);
const knip = JSON.parse(readFileSync(resolve(root, "knip.json"), "utf8")) as {
  workspaces: Record<string, { project: string[] }>;
};
const patterns = knip.workspaces["apps/grassmoth-client"].project;

function productionFiles(): Set<string> {
  const included = new Set<string>();
  const excluded = new Set<string>();
  for (const configured of patterns) {
    const negative = configured.startsWith("!");
    const pattern = (negative ? configured.slice(1) : configured).replace(/!$/, "");
    const target = negative ? excluded : included;
    for (const file of new Bun.Glob(pattern).scanSync({ cwd: web, absolute: true, onlyFiles: true })) {
      target.add(resolve(file));
    }
  }
  return new Set([...included].filter((file) => !excluded.has(file)));
}

const production = productionFiles();
const files = config.fileNames.filter((file) => production.has(resolve(file)));
const generated = resolve(root, "packages/api-types/src/generated");
files.push(...ts.sys.readDirectory(generated, [".ts"], undefined, ["**/*.ts"]));
const host: ts.LanguageServiceHost = {
  getScriptFileNames: () => files,
  getScriptVersion: () => "0",
  getScriptSnapshot: (file) => {
    const source = ts.sys.readFile(file);
    return source === undefined ? undefined : ts.ScriptSnapshot.fromString(source);
  },
  getCurrentDirectory: () => web,
  getCompilationSettings: () => config.options,
  getDefaultLibFileName: (options) => ts.getDefaultLibFilePath(options),
  fileExists: ts.sys.fileExists,
  readFile: ts.sys.readFile,
  readDirectory: ts.sys.readDirectory,
  directoryExists: ts.sys.directoryExists,
  getDirectories: ts.sys.getDirectories,
  realpath: ts.sys.realpath,
};
const service = ts.createLanguageService(host);
const program = service.getProgram();
if (!program) throw new Error("TypeScript language service did not create the web program");

function name(node: ts.Node | undefined): string | undefined {
  return node && (ts.isIdentifier(node) || ts.isStringLiteralLike(node)) ? node.text : undefined;
}

function declarations(node: ts.Node): string[] {
  const owners: string[] = [];
  for (let parent = node.parent; parent; parent = parent.parent) {
    if (ts.isTypeAliasDeclaration(parent) || ts.isInterfaceDeclaration(parent)) owners.push(parent.name.text);
    if (ts.isPropertySignature(parent)) {
      const property = name(parent.name);
      if (property) owners.push(property);
    }
  }
  return owners;
}

function nodeAt(source: ts.SourceFile, position: number): ts.Node {
  let result: ts.Node = source;
  function visit(node: ts.Node): void {
    if (node.getStart(source) <= position && position < node.getEnd()) {
      result = node;
      ts.forEachChild(node, visit);
    }
  }
  visit(source);
  return result;
}

function reader(node: ts.Node): boolean {
  const parent = node.parent;
  if (ts.isPropertyAccessExpression(parent) && parent.name === node) {
    const outer = parent.parent;
    return !writes(outer, parent);
  }
  if (ts.isElementAccessExpression(parent) && parent.argumentExpression === node) {
    const outer = parent.parent;
    return !writes(outer, parent);
  }
  if (ts.isBindingElement(parent) && ts.isObjectBindingPattern(parent.parent)) return true;
  return false;
}

function writes(parent: ts.Node, expression: ts.Expression): boolean {
  if (ts.isBinaryExpression(parent) && parent.left === expression) {
    const operator = parent.operatorToken.kind;
    return operator >= ts.SyntaxKind.FirstAssignment && operator <= ts.SyntaxKind.LastAssignment;
  }
  return (ts.isPrefixUnaryExpression(parent) || ts.isPostfixUnaryExpression(parent)) &&
    (parent.operator === ts.SyntaxKind.PlusPlusToken || parent.operator === ts.SyntaxKind.MinusMinusToken);
}

const requested = JSON.parse(await Bun.stdin.text()) as ExportedField[];
const byAlias = new Map<string, ExportedField[]>();
for (const field of requested) {
  for (const alias of field.aliases) {
    const owners = byAlias.get(alias) ?? [];
    owners.push(field);
    byAlias.set(alias, owners);
  }
}
const consumed: Record<string, string> = {};
const opaque: Record<string, string> = {};
const declarationFields = new Map<ts.Declaration, string[]>();
const checker = program.getTypeChecker();
console.error(`web fields: ${production.size} production files`);
for (const source of program.getSourceFiles()) {
  if (!resolve(source.fileName).startsWith(`${generated}/`)) continue;
  function visit(node: ts.Node): void {
    if (ts.isPropertySignature(node)) {
      const alias = name(node.name);
      if (alias) {
        const owners = declarations(node);
        const candidates = (byAlias.get(alias) ?? []).filter(
          (field) => owners.includes(field.owner) || field.components.some((component) => owners.includes(component)),
        );
        if (candidates.length > 0) {
          declarationFields.set(node, candidates.map((field) => field.identity));
          const groups = service.findReferences(source.fileName, node.name.getStart(source)) ?? [];
          for (const group of groups) {
            for (const reference of group.references) {
              if (!production.has(resolve(reference.fileName)) || reference.isDefinition || reference.isWriteAccess) continue;
              const file = program.getSourceFile(reference.fileName);
              if (!file || !reader(nodeAt(file, reference.textSpan.start))) continue;
              const line = file.getLineAndCharacterOfPosition(reference.textSpan.start).line + 1;
              for (const field of candidates) consumed[field.identity] = `${reference.fileName}:${line}`;
            }
          }
        }
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
}

function record(symbol: ts.Symbol, target: Record<string, string>, source: ts.SourceFile, node: ts.Node): void {
  const line = source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1;
  for (const rootSymbol of checker.getRootSymbols(symbol)) {
    for (const declaration of rootSymbol.getDeclarations() ?? []) {
      for (const identity of declarationFields.get(declaration) ?? []) target[identity] = `${source.fileName}:${line}`;
    }
  }
}

function wholeObject(node: ts.Expression, source: ts.SourceFile): void {
  const pending = [checker.getTypeAtLocation(node)];
  const visited = new Set<ts.Type>();
  while (pending.length > 0) {
    const type = pending.pop()!;
    if (visited.has(type)) continue;
    visited.add(type);
    if (type.isUnionOrIntersection()) pending.push(...type.types);
    if (checker.isArrayType(type) || checker.isTupleType(type)) {
      pending.push(...checker.getTypeArguments(type as ts.TypeReference));
      continue;
    }
    for (const property of type.getProperties()) {
      const declaration = property.valueDeclaration ?? property.getDeclarations()?.[0];
      const belongsToAPI = checker.getRootSymbols(property).some((symbol) =>
        symbol.getDeclarations()?.some((definition) => declarationFields.has(definition)),
      );
      if (!belongsToAPI) continue;
      record(property, opaque, source, node);
      if (declaration) pending.push(checker.getTypeOfSymbolAtLocation(property, declaration));
    }
    const indexed = type.getStringIndexType();
    if (indexed) pending.push(indexed);
  }
}

function erasesProperties(type: ts.Type, visited = new Set<ts.Type>()): boolean {
  if (visited.has(type)) return false;
  visited.add(type);
  if (type.flags & (ts.TypeFlags.Any | ts.TypeFlags.Unknown)) return true;
  if (type.isUnion()) return type.types.some((member) => erasesProperties(member, visited));
  if (checker.isArrayType(type) || checker.isTupleType(type)) {
    return checker.getTypeArguments(type as ts.TypeReference).some((member) => erasesProperties(member, visited));
  }
  return type.getProperties().length === 0 && type.getStringIndexType() !== undefined;
}

function erasesOwner(actual: ts.Type, expected: ts.Type, visited = new Set<ts.Type>()): boolean {
  if (visited.has(actual)) return false;
  visited.add(actual);
  if (actual.isUnionOrIntersection()) {
    return actual.types.some((member) => erasesOwner(member, expected, visited));
  }
  if (checker.isArrayType(actual) && checker.isArrayType(expected)) {
    const [item] = checker.getTypeArguments(actual as ts.TypeReference);
    const [target] = checker.getTypeArguments(expected as ts.TypeReference);
    return Boolean(item && target && erasesOwner(item, target, visited));
  }
  for (const property of actual.getProperties()) {
    const origins = checker.getRootSymbols(property).flatMap((symbol) => symbol.getDeclarations() ?? []);
    if (!origins.some((declaration) => declarationFields.has(declaration))) continue;
    const target = expected.getProperty(property.name);
    if (!target) continue;
    const targetOrigins = checker.getRootSymbols(target).flatMap((symbol) => symbol.getDeclarations() ?? []);
    if (!targetOrigins.some((declaration) => origins.includes(declaration))) return true;
  }
  return false;
}

for (const file of production) {
  const source = program.getSourceFile(file);
  if (!source) continue;
  function visit(node: ts.Node): void {
    if ((ts.isIdentifier(node) || ts.isStringLiteralLike(node)) && reader(node)) {
      const symbol = checker.getSymbolAtLocation(node);
      if (symbol) record(symbol, consumed, source!, node);
    }
    if (ts.isElementAccessExpression(node) && !ts.isStringLiteralLike(node.argumentExpression)) wholeObject(node.expression, source!);
    if (ts.isSpreadAssignment(node)) wholeObject(node.expression, source!);
    if (ts.isJsxSpreadAttribute(node)) wholeObject(node.expression, source!);
    if (ts.isJsxAttribute(node) && node.initializer && ts.isJsxExpression(node.initializer) && node.initializer.expression) {
      const expression = node.initializer.expression;
      const context = checker.getContextualType(expression);
      if (context && (erasesProperties(context) || erasesOwner(checker.getTypeAtLocation(expression), context))) {
        wholeObject(expression, source!);
      }
    }
    if (ts.isCallExpression(node)) {
      const signature = checker.getResolvedSignature(node);
      for (const [index, argument] of node.arguments.entries()) {
        const parameter = signature?.parameters[index];
        if (parameter) {
          const expected = checker.getTypeOfSymbolAtLocation(parameter, node);
          if (erasesProperties(expected) || erasesOwner(checker.getTypeAtLocation(argument), expected)) {
            wholeObject(argument, source!);
          }
        }
      }
    }
    if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression)) {
      const owner = node.expression.expression;
      if (ts.isIdentifier(owner) && owner.text === "Object" && ["entries", "values", "keys"].includes(node.expression.name.text)) {
        for (const argument of node.arguments) wholeObject(argument, source!);
      }
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
}
service.dispose();
process.stdout.write(JSON.stringify({ readers: consumed, opaque }));
