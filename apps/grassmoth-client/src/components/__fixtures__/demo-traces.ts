import latentStructureTrace from "../../../../../data/HEALTHDEMO/fixture/traces/latent_structure.json";
import measurementStructureTrace from "../../../../../data/HEALTHDEMO/fixture/traces/measurement_structure.json";
import measurementsTrace from "../../../../../data/HEALTHDEMO/fixture/traces/measurements.json";
import rawDataTrace from "../../../../../data/HEALTHDEMO/fixture/traces/raw_data.json";
import statisticalModelSpecTrace from "../../../../../data/HEALTHDEMO/fixture/traces/statistical_model_spec.json";

export const demoTraces = {
  raw_data: rawDataTrace,
  latent_structure: latentStructureTrace,
  measurement_structure: measurementStructureTrace,
  measurements: measurementsTrace,
  statistical_model_spec: statisticalModelSpecTrace,
} as const;
