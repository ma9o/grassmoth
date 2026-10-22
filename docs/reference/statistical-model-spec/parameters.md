# Parameters and Priors

The library owns the parameter and prior definitions:

- [Parameter specifications and transforms](../../../packages/grassmoth/src/grassmoth/artifacts/model/parameter_spec.py) define uncertain quantities, their law membership, and their authored scales.
- [Coefficient expressions](../../../packages/grassmoth/src/grassmoth/artifacts/model/expressions.py) define the roles and meanings of the slots that reference parameters.
- [Prior authoring](../../../packages/grassmoth/src/grassmoth/artifacts/probability/native.py) defines the supported constructors, argument vocabulary, and constraints.
- [Probability-law serialization](../../../packages/grassmoth/src/grassmoth/artifacts/probability/serialization.py) defines the saved representation of native laws.
- [Prior compilation](../../../packages/grassmoth/src/grassmoth/models/ssm/compile/prior_compilation.py) applies the authored transforms to complete probability laws.

See [assumptions and limits](../../assumptions.md) for the modeling commitments.
