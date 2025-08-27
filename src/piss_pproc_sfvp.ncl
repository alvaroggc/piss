;
; info: calculate stream function with CESM2 outputs
; author(ess): @alvaroggc
;
; usage: ncl x=<sim_id> piss_pproc_sfvp.ncl


; ##################
; ### PARAMETERS ###
; ##################


; directories
dirin  = "/mnt/cirrus/results/friquelme/data/piss/yData/h1/"
dirout = "/mnt/arcus/results/agomez/projects/piss/data/sfvp/"

; simulation ID (represented by variable 'x')
;x = "lgm_100"

; array with all modeled years
n = 35
years = new(n, string)

do i = 0, n - 1
    years(i) = sprintf("%04.0f", i+1)
end do

; input files (full paths)
fin = dirin + x + "/" + x + ".h1." + years + ".nc"

; netcdf output file (full path)
fout = dirout + x + ".h1.sfvp.nc"


; ###############
; ### RUNNING ###
; ###############


; logging message
print("> loading experiment " + x)

; remove output file (if exists)
system("/bin/rm -f " + fout)

; load files
ds   = addfiles(fin, "r")
sfvp = addfile(fout, "c")

; declare definition mode in output file (improves efficiency)
setfileoption(sfvp, "DefineMode", True)

; logging message
print("> separating variables")

; extract variables
time = ds[:]->time
lat  = ds[0]->lat
lon  = ds[0]->lon
U200 = ds[:]->U200
V200 = ds[:]->V200
U850 = ds[:]->U850
V850 = ds[:]->V850

; dimensions
nt   = dimsizes(time)
nlat = dimsizes(lat)
nlon = dimsizes(lon)

; predefine the coordinate variables and their dimensionality in output file
dimNames = (/"time", "lat", "lon"/)
dimSizes = (/ -1   ,  nlat,  nlon/)
dimUnlim = (/ True , False, False/)
filedimdef(sfvp, dimNames, dimSizes, dimUnlim)

; registro
print("> calculating stream functions and potential velocity")

; creación de arreglos para función corriente (SF) y velocidad potencial (VP)
SF200 = new((/nt, nlat, nlon/), double)
VP200 = new((/nt, nlat, nlon/), double)
SF850 = new((/nt, nlat, nlon/), double)
VP850 = new((/nt, nlat, nlon/), double)

; cálculo de sf y vp
uv2sfvpf(U200, V200, SF200, VP200)
uv2sfvpf(U850, V850, SF850, VP850)

; registro
print("> setting attributes to new variables")

; añadir atributos a SF y VP
SF200@units = "m**2/s"
VP200@units = "m**2/s"
SF200@long_name = "Stream function at 200 hPa"
VP200@long_name = "Velocity potential at 200 hPa"

SF850@units = "m**2/s"
VP850@units = "m**2/s"
SF850@long_name = "Stream function at 850 hPa"
VP850@long_name = "Velocity potential at 850 hPa"

; añadir dimensiones a SF y VP
SF200!0 = "time"
VP200!0 = "time"
SF200!1 = "lat"
VP200!1 = "lat"
SF200!2 = "lon"
VP200!2 = "lon"

SF850!0 = "time"
VP850!0 = "time"
SF850!1 = "lat"
VP850!1 = "lat"
SF850!2 = "lon"
VP850!2 = "lon"

; definición de nuevas variables a escribir
filevardef(sfvp, "time", typeof(time), getvardims(time))
filevardef(sfvp, "lat",  typeof(lat),  getvardims(lat) )
filevardef(sfvp, "lon",  typeof(lon),  getvardims(lon) )

filevardef(sfvp, "SF200", typeof(SF200), getvardims(SF200))
filevardef(sfvp, "VP200", typeof(VP200), getvardims(VP200))

filevardef(sfvp, "SF850", typeof(SF850), getvardims(SF850))
filevardef(sfvp, "VP850", typeof(VP850), getvardims(VP850))

; copia de nuevos atributos
filevarattdef(sfvp, "time", time)
filevarattdef(sfvp, "lat",  lat )
filevarattdef(sfvp, "lon",  lon )

filevarattdef(sfvp, "SF200", SF200)
filevarattdef(sfvp, "VP200", VP200)

filevarattdef(sfvp, "SF850", SF850)
filevarattdef(sfvp, "VP850", VP850)

;==============================================================================
; ESCRIBIENDO NUEVAS VARIABLES A ARCHIVO ORIGINAL
;==============================================================================

; registro
print("> writing new variables to output file")

; explicitly exit file definition mode
setfileoption(sfvp, "DefineMode", False)

; escritura de nuevas variables
sfvp->time  = (/time/)
sfvp->lat   = (/lat/)
sfvp->lon   = (/lon/)

sfvp->SF200 = (/SF200/)
sfvp->VP200 = (/VP200/)

sfvp->SF850 = (/SF850/)
sfvp->VP850 = (/VP850/)

; registro final
print(" ")









