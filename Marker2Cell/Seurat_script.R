library(dplyr)
library(Seurat)
library(patchwork)

# First the location of the dataset is specified

scdata_dir<-"./filtered_gene_bc_matrices/hg19"

# sc.data is read in:
sc.data <-Read10X(data.dir=scdata_dir)


# the loop loops through every single cell in the dataset and creates a csv file for each cell
# the csv file contains which RNA was counted how many times in descending order
for (i in 1:ncol(sc.data)) {
  which_cell<-i

  sc <- CreateSeuratObject(counts=sc.data, project="sc", min.cells =3 , min.features = 200)
  
  data_one_cell<-sort(subset(sc.data[,which_cell],sc.data[,which_cell]>0), decreasing = TRUE)
  write.csv(data_one_cell,file=paste(scdata_dir,"_cell_",i))
}


