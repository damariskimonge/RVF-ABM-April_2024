setwd("/Users/damariskimonge/Documents/python damaris/RVF-ABM-April_2024")
#load packages
library(pacman)
p_load(sf)
p_load(terra)
p_load(raster)


p_load(readr)
p_load(readxl)
p_load(haven)
p_load(dplyr)
p_load(stringr)
p_load(mapview)
p_load(writexl)


#load shapefile
district_shp<-st_read("data/shapefile/uga_admbnda_ubos_20200824_shp/uga_admbnda_adm2_ubos_20200824.shp") |>
  distinct(ADM2_EN, .keep_all = TRUE)

uganda1 <- st_read("data/shapefile/uga_admbnda_ubos_20200824_shp/uga_admbnda_adm0_ubos_20200824.shp") # as a sf
uganda <- vect("data/shapefile/uga_admbnda_ubos_20200824_shp/uga_admbnda_adm0_ubos_20200824.shp") # as a spatial vector

#load raster files
ndvi<-rast("data/ndvi_uganda2021.tif")

apr_2020<-rast("data/temp/apr_2020.tif")
apr_2021<-rast("data/temp/apr_2021.tif")
aug_2020<-rast("data/temp/aug_2020.tif")
aug_2021<-rast("data/temp/aug_2021.tif")
dec_2020<-rast("data/temp/dec_2020.tif")
dec_2021<-rast("data/temp/dec_2021.tif")
feb_2020<-rast("data/temp/feb_2020.tif")
feb_2021<-rast("data/temp/feb_2021.tif")
jan_2020<-rast("data/temp/jan_2020.tif")
jan_2021<-rast("data/temp/jan_2021.tif")
jul_2020<-rast("data/temp/jul_2020.tif")
jul_2021<-rast("data/temp/jul_2021.tif")
jun_2020<-rast("data/temp/jun_2020.tif")
jun_2021<-rast("data/temp/jun_2021.tif")
mar_2020<-rast("data/temp/mar_2020.tif")
mar_2021<-rast("data/temp/mar_2021.tif")
may_2020<-rast("data/temp/may_2020.tif")
may_2021<-rast("data/temp/may_2021.tif")
nov_2020<-rast("data/temp/nov_2020.tif")
nov_2021<-rast("data/temp/nov_2021.tif")
oct_2020<-rast("data/temp/oct_2020.tif")
oct_2021<-rast("data/temp/oct_2021.tif")
sep_2020<-rast("data/temp/sep_2020.tif")
sep_2021<-rast("data/temp/sep_2021.tif")

precip_2021<-rast("data/precip_uganda2021.tiff")
#resample to 30m*30m
grid_prec <- rast(ext(precip_2021))
res(grid_prec) <- 0.000269
crs(grid_prec) <- crs(precip_2021)
prec_resample <- resample(precip_2021, grid_prec, "near")


grid_ndvi <- rast(ext(ndvi))
res(grid_ndvi) <- 0.000269
crs(grid_ndvi) <- crs(ndvi)
ndvi_resample <- resample(ndvi, grid_ndvi, "near")

temp_2021<-rast("data/precip_uganda2021.tiff")
#temp_rep<-project(temp_2021, district_shp)



require(raster)

ugs <- list(
  ndvi,
  apr_2020,
  apr_2021,
  aug_2020,
  aug_2021,
  dec_2020,
  dec_2021,
  feb_2020,
  feb_2021,
  jan_2020,
  jan_2021,
  jul_2020,
  jul_2021,
  jun_2020,
  jun_2021,
  mar_2020,
  mar_2021,
  may_2020,
  may_2021,
  nov_2020,
  nov_2021,
  oct_2020,
  oct_2021,
  sep_2020,
  sep_2021
)

require(purrr)

temp_ug <-map(ugs,
              ~extract(.x, district_shp, fun = mean, na.rm =T)) %>% 
  setNames(c(
    
    'ndvi',
    'apr_2020',
    'apr_2021',
    'aug_2020',
    'aug_2021',
    'dec_2020',
    'dec_2021',
    'feb_2020',
    'feb_2021',
    'jan_2020',
    'jan_2021',
    'jul_2020',
    'jul_2021',
    'jun_2020',
    'jun_2021',
    'mar_2020',
    'mar_2021',
    'may_2020',
    'may_2021',
    'nov_2020',
    'nov_2021',
    'oct_2020',
    'oct_2021',
    'sep_2020',
    'sep_2021'
  ))

jj <- map2(temp_ug[2:24], names(temp_ug[2:24]),
           ~cbind(district_shp$ADM2_EN, .x) %>% 
             data.frame() %>% 
             dplyr::select(-ID) %>% 
             setNames(c('district', 'temp',)) %>% 
             mutate(date = .y)
) %>% 
  rbindlist()



# 
# ugs1 <- list(
#   ndvi_resample,
#   temp_2021,
#   prec_resample
# )

ugs1 <- list(
  ndvi,
  temp_2021,
  precip_2021
)
require(purrr)

temp_ug1 <-map(ugs1,
              ~extract(.x, district_shp, fun = mean, na.rm =T)) %>% 
  setNames(c(
    
    'ndvi',
    'temp_2021',
    'prec_2021'
  ))

jj1 <- map2(temp_ug1[1:3], names(temp_ug1[1:3]),
           ~cbind(district_shp$ADM2_EN, .x) %>% 
             data.frame() %>% 
             dplyr::select(-ID)
             #setNames(c('district', 'ndvi','temp','precip')) %>% 
           
) %>% 
  reduce(merge, by = 'district_shp.ADM2_EN')%>%
  set_names("district", "vegetation_index", "temperature", "precipitation")%>%
  mutate(temperature_clean=((temperature*0.00341802)+149)-273.15)%>%
  dplyr::select(-temperature_clean)

  
env_data<-jj1

temp_2021<-rast("data/precip_uganda2021.tiff")
writexl::write_xlsx(env_data, "data/env_data.xlsx")
