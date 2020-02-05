setwd("~/Desktop/Papers/Course/Prog Lab B/")
result <- read.csv("tiles_rois/dataset.csv")
count <- read.table("centroids.count", header = F, sep = '\t')
a <- paste0(result$tile_name, ".csv")
result$count <- count$V2[match(a, count$V1)]
write.csv(result, file = "tiles.csv", col.names = F)
