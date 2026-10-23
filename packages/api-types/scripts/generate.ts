/** Generate operations, named contracts and generic declarations from one OpenAPI AST. */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import openapiTS, { astToString } from "openapi-typescript";
import ts from "typescript";

// biome-ignore lint/suspicious/noExplicitAny: JSON Schema extensions contain heterogeneous nodes.
type JsonSchema = any;
const ROOT = resolve(import.meta.dirname, "..");
const checkOnly = process.argv.includes("--check");
const changedPaths: string[] = [];

function writeOrCheck(path: string, content: string): void {
  if (checkOnly) {
    if (readFileSync(path, "utf8") !== content) changedPaths.push(relative(ROOT, path));
  } else {
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, content);
  }
}

const schema = JSON.parse(readFileSync(resolve(ROOT, "schemas/openapi.json"), "utf8"));
// The viewer consumes the binary representation. Keep its existing operation
// types while the published OpenAPI also describes JSON for curl callers.
for (const methods of Object.values(schema.paths) as Record<string, JsonSchema>[]) {
  for (const operation of Object.values(methods)) {
    const content = operation.responses?.["200"]?.content;
    if (content?.["application/msgpack"]) delete content["application/json"];
  }
}
const definitions: Record<string, JsonSchema> = schema.components.schemas;
const refTail = (ref: string) => ref.slice(ref.lastIndexOf("/") + 1);
const templates = new Map<string, string>(
  Object.entries(schema["x-typescript-generics"]).map(([name, ref]) => [
    refTail((ref as { $ref: string }).$ref),
    name,
  ]),
);
const live = new Set<string>();
function collectReferences(value: JsonSchema): void {
  if (!value || typeof value !== "object") return;
  if (typeof value.$ref === "string" && value.$ref.startsWith("#/components/schemas/")) {
    const name = refTail(value.$ref);
    if (!live.has(name)) {
      live.add(name);
      collectReferences(definitions[name]);
    }
  }
  for (const child of Object.values(value)) collectReferences(child);
}
collectReferences(schema.paths);
collectReferences(schema["x-contract-roots"]);
const outputs = new Map<string, string>();
for (const name of Object.keys(definitions)) {
  if (
    !live.has(name) ||
    templates.has(name) ||
    name.endsWith("-Input") ||
    definitions[name]["x-typescript-mode"] === "validation"
  )
    continue;
  const canonical = name.replace(/-Output$/, "");
  if (definitions[name]["x-python-module"] && !definitions[name]["x-typescript-type"]) {
    outputs.set(name, canonical);
  }
}
const component = (name: string) =>
  ts.factory.createIndexedAccessTypeNode(
    ts.factory.createIndexedAccessTypeNode(
      ts.factory.createTypeReferenceNode("components"),
      ts.factory.createLiteralTypeNode(ts.factory.createStringLiteral("schemas")),
    ),
    ts.factory.createLiteralTypeNode(ts.factory.createStringLiteral(name)),
  );

/** Parse only the explicit Python-owned operand/template extension. */
function operand(source: string, qualified = false): ts.TypeNode {
  const file = ts.createSourceFile(
    "operand.ts",
    `type Operand = ${source};`,
    ts.ScriptTarget.Latest,
    true,
  );
  const declaration = file.statements[0];
  if (!declaration || !ts.isTypeAliasDeclaration(declaration))
    throw new Error(`Invalid operand: ${source}`);
  return rewrite(declaration.type, (node) =>
    ts.isStringLiteral(node)
      ? ts.factory.createStringLiteral(node.text)
      : ts.isNumericLiteral(node)
        ? ts.factory.createNumericLiteral(node.text)
        : ts.isTemplateLiteralTypeNode(node)
          ? ts.factory.createTemplateLiteralType(
              ts.factory.createTemplateHead(node.head.text, node.head.rawText),
              node.templateSpans.map((span) =>
                ts.factory.createTemplateLiteralTypeSpan(
                  span.type,
                  ts.isTemplateTail(span.literal)
                    ? ts.factory.createTemplateTail(span.literal.text, span.literal.rawText)
                    : ts.factory.createTemplateMiddle(span.literal.text, span.literal.rawText),
                ),
              ),
            )
          : qualified && ts.isTypeReferenceNode(node) && ts.isIdentifier(node.typeName)
            ? ts.factory.updateTypeReferenceNode(
                node,
                ts.factory.createQualifiedName(
                  ts.factory.createIdentifier("Domain"),
                  node.typeName,
                ),
                node.typeArguments,
              )
            : node,
  );
}

function rewrite<T extends ts.Node>(node: T, replace: (node: ts.Node) => ts.Node): T {
  const result = ts.transform(node, [
    (context) => {
      const visit: ts.Visitor = (child) => replace(ts.visitEachChild(child, visit, context));
      return (root) => ts.visitNode(root, visit) as T;
    },
  ]);
  const [rewritten] = result.transformed;
  if (!rewritten) throw new Error("Compiler produced no transformed declaration");
  result.dispose();
  return rewritten;
}

function referenceName(node: ts.Node): string | undefined {
  if (
    !ts.isIndexedAccessTypeNode(node) ||
    !ts.isLiteralTypeNode(node.indexType) ||
    !ts.isStringLiteral(node.indexType.literal) ||
    !ts.isIndexedAccessTypeNode(node.objectType)
  )
    return;
  const owner = node.objectType;
  if (
    ts.isTypeReferenceNode(owner.objectType) &&
    ts.isIdentifier(owner.objectType.typeName) &&
    owner.objectType.typeName.text === "components" &&
    ts.isLiteralTypeNode(owner.indexType) &&
    ts.isStringLiteral(owner.indexType.literal) &&
    owner.indexType.literal.text === "schemas"
  ) {
    return node.indexType.literal.text;
  }
}

const ast = await openapiTS(schema, {
  immutable: true,
  defaultNonNullable: false,
  inject: 'import type * as Domain from "./models";',
  transform(value) {
    if (typeof value.tsType === "string") return operand(value.tsType);
    // Named map aliases are schema roots, so transformProperty never sees them.
    const keyRef = (value.propertyNames as { $ref?: string } | undefined)?.$ref;
    const valueRef = (value.additionalProperties as { $ref?: string } | undefined)?.$ref;
    if (keyRef && valueRef) {
      return ts.factory.createTypeReferenceNode("Readonly", [
        ts.factory.createTypeReferenceNode("Partial", [
          ts.factory.createTypeReferenceNode("Record", [
            component(refTail(keyRef)),
            component(refTail(valueRef)),
          ]),
        ]),
      ]);
    }
    if (
      value.type === "string" &&
      (value.format === "binary" || value.contentMediaType === "application/octet-stream")
    ) {
      return ts.factory.createTypeReferenceNode("Blob");
    }
  },
  transformProperty(property, value) {
    // Preserve an authored minimum on open-ended arrays using the compiled item
    // type. Native prefix-item tuples keep their original readonly shape.
    if (
      value.type === "array" &&
      value.minItems > 0 &&
      value.maxItems === undefined &&
      property.type &&
      ts.isTypeOperatorNode(property.type) &&
      ts.isArrayTypeNode(property.type.type)
    ) {
      const item = property.type.type.elementType;
      return ts.factory.updatePropertySignature(
        property,
        property.modifiers,
        property.name,
        property.questionToken,
        ts.factory.createTypeOperatorNode(
          ts.SyntaxKind.ReadonlyKeyword,
          ts.factory.createTupleTypeNode([
            ...Array.from({ length: value.minItems }, () => item),
            ts.factory.createRestTypeNode(ts.factory.createArrayTypeNode(item)),
          ]),
        ),
      );
    }
    const keyRef = (value.propertyNames as { $ref?: string } | undefined)?.$ref;
    if (keyRef && property.type && ts.isTypeLiteralNode(property.type)) {
      const index = property.type.members.find(ts.isIndexSignatureDeclaration);
      if (index) {
        const map = ts.factory.createTypeReferenceNode("Readonly", [
          ts.factory.createTypeReferenceNode("Partial", [
            ts.factory.createTypeReferenceNode("Record", [component(refTail(keyRef)), index.type]),
          ]),
        ]);
        return ts.factory.updatePropertySignature(
          property,
          property.modifiers,
          property.name,
          property.questionToken,
          map,
        );
      }
    }
  },
});
const declarations = ast.find(
  (node) => ts.isInterfaceDeclaration(node) && node.name.text === "components",
);
if (!declarations || !ts.isInterfaceDeclaration(declarations))
  throw new Error("Missing component declarations");
// Compiler nodes are synthetic: inspect their names, not source positions.
const schemaMember = declarations.members.find(
  (member) =>
    ts.isPropertySignature(member) &&
    (ts.isIdentifier(member.name) || ts.isStringLiteral(member.name)) &&
    member.name.text === "schemas",
);
if (
  !schemaMember ||
  !ts.isPropertySignature(schemaMember) ||
  !schemaMember.type ||
  !ts.isTypeLiteralNode(schemaMember.type)
) {
  throw new Error("Missing component schema members");
}
const bodies = new Map(
  schemaMember.type.members.filter(ts.isPropertySignature).map((member) => {
    if (!member.type) throw new Error("Compiler produced an untyped schema member");
    return [(member.name as ts.Identifier | ts.StringLiteral).text, member.type] as const;
  }),
);
function schemaBody(name: string): ts.TypeNode {
  const body = bodies.get(name);
  if (!body) throw new Error(`Compiler omitted schema ${name}`);
  return body;
}
// Transport aliases are recursive in both modes. Direct named bodies break
// the indexed component/property recursion that TypeScript rejects (TS2502).
const transportAliases = new Map(
  Object.entries(definitions)
    .filter(([, value]) =>
      ["grassmoth.json_types", "grassmoth.artifacts.probability.serialization"].includes(
        value["x-python-module"],
      ),
    )
    .map(([name]) => [name, name.replace(/-(Input|Output)$/, "")]),
);
const named = [...outputs].map(([name, canonical]) =>
  ts.factory.createTypeAliasDeclaration(
    [ts.factory.createModifier(ts.SyntaxKind.ExportKeyword)],
    canonical,
    undefined,
    transportAliases.has(name)
      ? rewrite(schemaBody(name), (node) => {
          const ref = referenceName(node);
          const alias = ref && transportAliases.get(ref);
          return alias ? ts.factory.createTypeReferenceNode(alias) : node;
        })
      : component(name),
  ),
);
for (const [name, canonical] of templates) {
  const body = rewrite(schemaBody(name), (node) => {
    const ref = referenceName(node);
    if (!ref) return node;
    const application = definitions[ref]["x-typescript-type"];
    if (application) return operand(application);
    const publicName = templates.get(ref) ?? outputs.get(ref);
    return publicName
      ? ts.factory.createTypeReferenceNode(
          publicName,
          templates.has(ref)
            ? definitions[ref]["x-typescript-parameters"].map((parameter: string) =>
                ts.factory.createTypeReferenceNode(parameter),
              )
            : undefined,
        )
      : node;
  });
  named.push(
    ts.factory.createTypeAliasDeclaration(
      [ts.factory.createModifier(ts.SyntaxKind.ExportKeyword)],
      canonical,
      definitions[name]["x-typescript-parameters"].map((parameter: string) =>
        ts.factory.createTypeParameterDeclaration(undefined, parameter),
      ),
      body,
    ),
  );
}

const operations = ast.map((node) =>
  rewrite(node, (child) => {
    if (
      ts.isPropertySignature(child) &&
      (ts.isIdentifier(child.name) || ts.isStringLiteral(child.name)) &&
      child.name.text === "schemas" &&
      child.type &&
      ts.isTypeLiteralNode(child.type)
    ) {
      return ts.factory.updatePropertySignature(
        child,
        child.modifiers,
        child.name,
        child.questionToken,
        ts.factory.updateTypeLiteralNode(
          child.type,
          child.type.members.filter(
            (member) =>
              ts.isPropertySignature(member) &&
              (ts.isIdentifier(member.name) || ts.isStringLiteral(member.name)) &&
              live.has(member.name.text),
          ),
        ),
      );
    }
    const name = referenceName(child);
    if (name && transportAliases.has(name)) return operand(`Domain.${transportAliases.get(name)}`);
    const application =
      name &&
      definitions[name]["x-typescript-mode"] === "serialization" &&
      definitions[name]["x-typescript-type"];
    return application ? operand(application, true) : child;
  }),
);
const banner = "/** AUTO-GENERATED from Python's OpenAPI graph. Run bun run codegen. */\n";
writeOrCheck(resolve(ROOT, "src/generated/model-api.ts"), banner + astToString(operations));
writeOrCheck(
  resolve(ROOT, "src/generated/models.ts"),
  `${banner}import type { components } from "./model-api";\n${astToString(named)}`,
);
if (changedPaths.length)
  throw new Error(`Generated API types are stale: ${changedPaths.join(", ")}`);
console.log(checkOnly ? "API types checked." : "API types generated from one OpenAPI AST.");
