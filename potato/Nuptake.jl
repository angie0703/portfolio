#=
Version: 1.0.0
Creator: Angela Losu
Creation Date: 2025-12-01
Code description:
This module defines the data structures and parameters specific to the potato species within the Virtual Plant Lab framework.

Credit to Jie Lu for the original wheat model code.
=#
module Nuptake

using VirtualPlantLab 
using Base.Threads: @threads
import Random
using FastGaussQuadrature
using Distributions
#Random.seed!(123456789)
using Distributions,Plots,ColorTypes
using Ecophys
using SkyDomes
using PlantGeomPrimitives
using DataFrames
using Parameters
using CSV
using DelimitedFiles
import GLMakie
import IterTools
import Parameters:@with_kw
# import PlantGeomPrimitives:@Mesh

#N fixation function
function Nfix!(vars,all_roots)
for r in all_roots
    if r.Nodule==true
    vars.nodulebiomass+=r.nodulebiomass
    end
end


vars.Nfix=max(0,(vars.nodulebiomass*1.21)*exp(-180*vars.Nc))
vars.Nup+=vars.Nfix
vars.totalN+=vars.Nfix
vars.accNfix+=vars.Nfix

#println("nodebiomass$(vars.nodulebiomass)")
#println("Nfix$(vars.accNfix)")
end

function exploresoil!(all_roots, all_rootmeristem ,all_lrootmeristem, all_soils,vars)
    
    for ss in all_soils
        #ss.Ni=ss.Nc*ss.length*ss.width*ss.height*1e3
    HAT=(vars.Imax*max(0,(ss.Ni-ss.Nm*ss.length*ss.width*ss.height*1e3))/(ss.length*ss.width*ss.height*1e3))/(vars.Km+max(0,(ss.Ni-ss.Nm*ss.length*ss.width*ss.height*1e3))/(ss.length*ss.width*ss.height*1e3))
    LAT=(vars.K2*max(0,ss.Ni-ss.Nm*ss.length*ss.width*ss.height*1e3)/((ss.length*ss.width*ss.height*1e3)))
    eN=exp(-80*vars.Nc)
  for r in all_roots
     if ss.x-ss.length/2<r.x<ss.x+ss.length/2 && ss.y-ss.width/2<r.y<ss.y+ss.width/2 && ss.z-ss.height/2<r.z<ss.z+ss.height/2
        if r.order==0        
        r.potNup=((pi*vars.width)*vars.IBD*HAT+LAT*r.biomass)*eN
        else
        r.potNup=((pi*vars.width*vars.RDM)*vars.IBD*HAT+LAT*r.biomass)*eN+(r.finerootbiomass*LAT+4*r.finerootbiomass/(vars.RTD*1e3*vars.finerootD)*HAT)*eN
        end
                vars.potNup+=r.potNup
                ss.potNup+=r.potNup
     end
     #println("order$(r.order)biomass$(r.biomass)")
    end 

          for r in all_roots    
             if ss.x-ss.length/2<r.x<ss.x+ss.length/2 && ss.y-ss.width/2<r.y<ss.y+ss.width/2 && ss.z-ss.height/2<r.z<ss.z+ss.height/2        
                r.actNup=min(r.potNup,max(0,(ss.Ni-ss.Nm))*r.potNup/ss.potNup)
                ss.Ni-=r.actNup
                ss.rl+=vars.IBD
                vars.Nup+=(r.actNup)*14/1e6
                vars.totalN+=r.actNup*14/1e6
                if ss.Nc<150
                r.Nodule=true
                end
       end 
    end

   
    #vars.totalN=vars.Nup+vars.N0
 ss.Nc=ss.Ni/(ss.height*ss.width*ss.length*1000)
   #### Update auxin distribution in lateral and axial root meristems based on soil N availability
    for lrmer in all_lrootmeristem
        if ss.x-ss.length/2<lrmer.x<ss.x+ss.length/2 && ss.y-ss.width/2<lrmer.y<ss.y+ss.width/2 && ss.z-ss.height/2<lrmer.z<ss.z+ss.height/2
            if vars.exNaux==true
                lrmer.decay=(1-min(1,0.001*ss.Ni))*0.43 #0.001
                #lrmer.merauxinN=lrmer.merauxin
                lrmer.NRTcoef=1 
                lrmer.merauxinN=(1-min(0.82,0.00082*ss.Ni))*vars.merauxin*vars.RDM*lrmer.NRTcoef  #0.00082
            else
                lrmer.merauxinN=vars.merauxin*vars.RDM*0.18
                vars.decay=0 
            end
        end
    end

    for rmer in all_rootmeristem
        #rmer.merauxinN=rmer.merauxin
        #println("x$(rmer.x)y$(rmer.y)z$(rmer.z)")
        if ss.x-ss.length/2<=rmer.x<=ss.x+ss.length/2 && ss.y-ss.width/2<=rmer.y<=ss.y+ss.width/2 && ss.z-ss.height/2<=rmer.z<=ss.z+ss.height/2
            
            if vars.exNaux==true     
            rmer.merauxcoef=(1-min(0.82,0.00082*ss.Ni))
            rmer.merauxinN=vars.merauxin*rmer.merauxcoef
            rmer.decay=(1-min(1,0.0010*ss.Ni))*0.43
            #println("merauxin$(vars.merauxin)")
            else
           rmer.merauxinN=vars.merauxin*0.18
           vars.decay=0
            end
        end
    end
  
end
vars.Nc=(vars.Nup+vars.N0)/(vars.actbiomass+vars.seedbiomass)
vars.allN=vars.Nup+vars.N0
println("$(vars.species)_Nc $(vars.Nc), Nup $(vars.Nup) actbiomass $(vars.actbiomass)")
end
end