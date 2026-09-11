from soda import q
for ds in ["qhpu-8ixx","djw7-ur7t"]:
    q(ds,"Q1 rango fechas",{"$select":"min(fecha_corte) as fmin, max(fecha_corte) as fmax, count(*) as n"})
    q(ds,"Q2 subtipos",{"$select":"subtipo_negocio, nombre_subtipo_patrimonio, count(*) as n","$group":"subtipo_negocio, nombre_subtipo_patrimonio","$order":"subtipo_negocio"})
    q(ds,"Q3 rows por año",{"$select":"date_trunc_y(fecha_corte) as anio, count(*) as n_rows, min(fecha_corte) as fmin, max(fecha_corte) as fmax","$group":"anio","$order":"anio"})
