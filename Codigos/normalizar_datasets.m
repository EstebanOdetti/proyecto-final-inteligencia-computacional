clc;clear all; close all;
pkg load io
datosTotales=xlsread("heart2.csv");
salida_esperada=datosTotales(2:end,end);
datosTotales=datosTotales(2:end,1:end-1);

for j=1:columns(datosTotales)
  maxium=max(datosTotales(:,j));
  minimun=min(datosTotales(:,j));
  for i=1:rows(datosTotales)
    datosTotales2(i,j)=(datosTotales(i,j)-minimun)/(maxium-minimun);
  endfor
endfor
datosTotales2=[datosTotales2 salida_esperada];
csvwrite("heart2_normalizado.csv",datosTotales2)