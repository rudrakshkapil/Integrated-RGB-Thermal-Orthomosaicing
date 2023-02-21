# Here I process raw data to produce thermal tiffs

library(terra)

# The fancier error catching  is required to prevent the while loop from terminating when 
# an expected raw data file is missing.
tryCatch.W.E <- function(expr) {
  W <- NULL
  w.handler <- function(w){ # warning handler
    W <<- w
    invokeRestart("muffleWarning")
  }
  list(value = withCallingHandlers(tryCatch(expr, error = function(e) e),
                                   warning = w.handler),
       warning = W)
}

# Running for the first series in the Fall
# initializing
args = commandArgs(trailingOnly=TRUE)



i = 1
max_val = strtoi(args[3])
prefix = args[1] # "D:/Cynthia_Data/2022_06_03/thermal_raw/img_"
out_prefix = args[2] # "D:/Cynthia_Data/2022_06_03/thermal_tiffs/img_"
# Initializes the progress bar
pb <- txtProgressBar(min = 0,      # Minimum value of the progress bar
                     max = max_val, # Maximum value of the progress bar
                     style = 3,    # Progress bar style (also available style = 1 and style = 2)
                     width = 50,   # Progress bar width. Defaults to getOption("width")
                     char = "=")   # Character used to create the bar

while(i <= max_val) {
  rawin <- paste(prefix, as.character(i - 1), ".raw", sep = "")
  if(is.numeric(tryCatch.W.E(readBin(rawin, integer(), n = 640*512, size = 2))$value) == TRUE) {
    r1 <- readBin(rawin, integer(), n = 640*512, size = 2)
    r1mat <- matrix(r1, nrow = 512, ncol = 640, byrow = TRUE)
    r1rast <- rast(r1mat)
    plot(r1rast)
    tiffout <- paste(out_prefix, as.character(i-1), ".tiff", sep = "")
    writeRaster(r1rast, filename = tiffout,  datatype = 'INT2U', gdal = "COMPRESS=NONE", overwrite = TRUE)
  }
  setTxtProgressBar(pb, i)
  i = i + 1
}
close(pb)