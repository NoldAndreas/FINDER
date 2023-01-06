library(shiny)
library(maps)
library(mapproj)
library("plotly")
library(pivottabler)
library(reshape2)

#******************************************************************************
#******* Read clustersizes ****************************************************
#******************************************************************************
processFile = function(filepath) {
  l<-c();
  con = file(filepath, "r")
  while ( TRUE ) {
    line = readLines(con, n = 1)
    if ( length(line) == 0 ) {
      break
    }
    arr <- lapply(strsplit((line), split=" "), function(x) (as.numeric(x)))
    l <- c(l,arr);
    #print(arr)
    #break;
  }
  close(con)
  return(l);
}
#******************************************************************************
#******************************************************************************
#******************************************************************************

foldername     <- "../../../MikeData/Output_R/";

data        <- read.csv(paste(foldername,"results_dataWindow_1.csv",sep=""), header = TRUE)
data_X      <- read.csv(paste(foldername,"dataWindow_1_filtered_signal.txt",sep=""),sep="\t",header = FALSE);
data_labels <- read.csv(paste(foldername,"dataWindow_1_labels_signal.txt",sep=""),header=FALSE,sep="\t");
data_clusterSizes <- read.csv(paste(foldername,"dataWindow_1_clusterSizes.txt",sep=""),header=TRUE,sep=",");
#data_clusterSizes <- processFile(paste(foldername,"dataWindow_1_clusterSizes.txt",sep=""))

data$sigma <- data$sigma %>% round(digits=3)

# Define UI for app that draws a histogram ----
ui <- fluidPage(

  # App title ----
  titlePanel("See clustering results"),

  # Sidebar layout with input and output definitions ----
  sidebarLayout(

    # Sidebar panel for inputs ----
    sidebarPanel(

      helpText("Seeing clustering result of",
               "some localization data"),
      # Input: Slider for the number of bins ----
      selectInput("select", h3("Choose a variable to display"),
                  choices = c("noise","signal"), selected = "noise"),

      sliderInput("sliderThreshold", h3("Threshold:"),
                  min = min(data$threshold), max = max(data$threshold), value = median(data$threshold)),

      sliderInput("sliderSigma", h3("Sigma:"),
                  min = round(min(data$sigma),digits=2), max = round(max(data$sigma),digits=2), value = median(round(data$sigma,digits = 2)))

    ),

    # Main panel for displaying outputs ----
    mainPanel(
      textOutput("selected_var"),
      textOutput("selected_range"),
      plotlyOutput("map"),
      textOutput("selectedClustering"),
      plotlyOutput("points"),
      plotlyOutput("clusterSizes"),
      textOutput("other")
    )
  )
)


# Define server logic required to draw a histogram ----
server <- function(input, output,session) {

  #******************************************************************************
  #Get index that gets closest to selected sigma and threshold
  data_sel <- reactive({
    if(input$select == "signal"){
      dataP <- data %>% filter(type=="signal");
    }
    else{
      dataP <- data %>% filter(type=="noise")
    }
    return(dataP);
  });
  idx      <- reactive({return(which.min(abs(input$sliderThreshold - data_sel()$threshold) + abs(input$sliderSigma - data_sel()$sigma)));});
  data_clusterSizes_sel <- reactive({
    data_clusterSizes %>% filter(index==idx()-1)
  })
  #******************************************************************************

  output$selected_var <- renderText({paste("You have selected ",input$select)})

  output$selectedClustering <- renderPrint({
    return(paste("Selected clustering: Threshold = ",data_sel()$threshold[idx()]," , sigma = ",round(data_sel()$sigma[idx()],digits=2),sep=''));
    })

  #******************************************************************************
  # Plot heatmap
  #******************************************************************************
  output$map <- renderPlotly({
      data_table <- acast(data_sel(), threshold~sigma, value.var="no_clusters",mean)
      plot_ly(x=colnames(data_table), y=rownames(data_table), z = data_table, type = "heatmap")
  })
  #******************************************************************************

  #******************************************************************************
  # Plot points
  #******************************************************************************
  output$points <- renderPlotly({
    labels <- data_labels[,idx()];
    fig <- plot_ly(data_X,x = ~V1, y = ~V2, type = 'scatter',size=~(1+5*(labels>=0)),width = 500, height = 500,color=1*(labels>=0),marker=list(size=3));
    fig
  })
  #******************************************************************************
  output$clusterSizes <- renderPlotly({
    fig <- plot_ly(x=data_clusterSizes_sel()$size,type = "histogram",bingroup=10)
  })

  output$other <- renderText({length(data_clusterSizes_sel()$size)});
}

# Create Shiny app ----
shinyApp(ui = ui, server = server)
