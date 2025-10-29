# 00_setup_paths.R
PROJECT_ROOT <- here::here()
DATA_PATH <- file.path(PROJECT_ROOT, "data", "tripadvisor.csv")
if(!file.exists(DATA_PATH)) stop("Dataset not found at: ", DATA_PATH)

OUTPUT_FIGURES <- file.path(PROJECT_ROOT, "figures")
OUTPUT_MODELS <- file.path(PROJECT_ROOT, "models")
dir.create(OUTPUT_FIGURES, showWarnings = FALSE)
dir.create(OUTPUT_MODELS, showWarnings = FALSE)

message("Project root: ", PROJECT_ROOT)
message("Data path: ", DATA_PATH)
