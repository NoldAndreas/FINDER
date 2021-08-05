library(pivottabler)
library(dplyr)
library("ggplot2") 

library(reshape2)

data_table <- acast(data, threshold~sigma, value.var="no_clusters", mean)
heatmap(data_table)

data <- read.csv("../../../MikeData/ReferenceCheck/results_dataWindow_1.csv", header = TRUE)
dataPivot <- data %>% filter(type == "signal") %>% qpvt("sigma", "threshold", "mean(no_clusters)") 


ggp <- ggplot(data, aes(sigma, threshold)) +                           # Create heatmap with ggplot2
  geom_tile(aes(fill = no_clusters))

qpvt(data, "sigma", "threshold", "mean(no_clusters)")

pt <- PivotTable$new()
pt$addData(data)
pt$addColumnDataGroups("sigma")
pt$addRowDataGroups("threshold")                #     **** LINE BELOW ADDED ****
pt$defineCalculation(calculationName="no_clusters", summariseExpression="n()")
pt$renderPivot()

qpvt(bhmtrains, "TOC", "TrainCategory", "mean(SchedSpeedMPH,na.rm=TRUE)")


#Checking filter 
starwars %>% filter(mass > mean(mass, na.rm = TRUE))
data %>% filter(sigam > 0.07)