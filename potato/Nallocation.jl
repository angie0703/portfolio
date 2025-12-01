#=
Version: 1.0.0
Creator: Angela Losu
Creation Date: 2025-12-01
Code description:
This module defines the data structures and parameters specific to the potato species within the Virtual Plant Lab framework.

Credit to Jie Lu for the original wheat model code.
=#
module Nallocation

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


##if photosynthesis is enabled, calculate the nitrogen distrubution within the plant
#N sink strength calculation
function N_sinkstrength!(species,all_leaves,all_internodes,all_petiole,all_grains)
vars=data(species)
 for l in all_leaves
     if l.fabs>0
     l.DLNarea=vars.LN0*l.fabs^vars.kNkl
     else
         l.DLNarea=vars.LN0
     end
    l.DLN=l.DLNarea*(l.area)
    vars.totalNsink+=l.DLN
   vars.totalNphotosink+=l.DLN
end

for pet in all_petiole
    if pet.fabs>0
    pet.DLNarea=vars.LN0*pet.fabs^vars.kNkl
    else
        pet.DLNarea=vars.LN0
    end
    pet.DLN=pet.DLNarea*pet.area
    vars.totalNsink+=pet.DLN
     vars.totalNphotosink+=pet.DLN
end

for int in all_internodes
      #nitrogen sink strength per internode per day(fixed fraction for structural internodes)
        int.Nsink=(int.biomass-int.biomassprevious)*vars.fN_internode
         
        #total nitrogen sink strength of a plant per day
        vars.totalNsinkstruct+=int.Nsink
        vars.totalNsink+=int.Nsink
    if int.fabs>0
    int.DLNarea=vars.LN0*int.fabs^vars.kNkl
    else
        int.DLNarea=vars.LN0
    end
    int.DLN=int.DLNarea*int.area
    vars.totalNsink+=int.DLN
    vars.totalNphotosink+=int.DLN
end
 for g in all_grains
#nitrogen sink strength per grain per day(fixed fraction for structural grains)
        g.Nsink=(g.biomass-g.biomassprevious)*vars.fN_grain
       
        #total nitrogen sink strength of a plant per day
        vars.totalNsinkstruct+=g.Nsink 
         vars.totalNsink+=g.Nsink
end
        
        #nitrogen sink strength per root system per day
        vars.rootNsink=vars.actbiomass_rootperday*vars.fN_root 
        #total nitrogen sink strength of a plant per day
        vars.totalNsinkstruct+=vars.rootNsink
        vars.totalNsink+=vars.rootNsink
        #println("totalNsink$(vars.totalNsink) rootNsink$(vars.rootNsink)")
end



function N_allocation!(all_leaves,all_petiole,all_internodes,all_grains,vars)

relativeNsinkstrength::Float64=0
##structural N and eliminate from the total N
for int in all_internodes
 if vars.totalNsink<=0
    relativeNsinkstrength=0
   else
    relativeNsinkstrength=int.Nsink/vars.totalNsink
   end
if vars.totalN>0
    rate=min(int.Nsink,vars.totalN*relativeNsinkstrength)
    vars.totalN= vars.totalN-rate 
    vars.intN+=rate
    else
    vars.totalN=0.0
    end
end
##structional N in grains
for g in all_grains
    if vars.totalNsink<=0
    relativeNsinkstrength=0
   else
    relativeNsinkstrength=g.Nsink/vars.totalNsink
   end

     if vars.totalN>0
        r=min(g.Nsink,vars.totalN*relativeNsinkstrength)
        vars.totalN= vars.totalN-r
        vars.grainN+=r
     else
        vars.totalN=0.0
    end
end

##structural N in root system
 if vars.totalNsink<=0
    relativeNsinkstrength=0
   else
    relativeNsinkstrength=vars.rootNsink/vars.totalNsink
   end

 if vars.totalN>0
  vars.totalN-=min(vars.rootNsink,vars.totalN*relativeNsinkstrength)
    vars.rootN+=min(vars.rootNsink,vars.totalN*relativeNsinkstrength)
 else
    vars.totalN=0.0
  end
##photosynthetic N allocation
for l in all_leaves
    vars.totalphotoarea+=l.area
   if vars.totalNsink<=0
    relativeNsinkstrength=0
   else
    relativeNsinkstrength=l.DLN/vars.totalNsink
   end
   if l.DLN>relativeNsinkstrength*(vars.totalN)
   l.Navaliable=relativeNsinkstrength*(vars.totalN)
   else
    l.Navaliable=l.DLN
   end
     l.LNarea = l.Navaliable/max(1e-8,l.area)
end 
for pet in all_petiole
    vars.totalphotoarea+=pet.area
   if vars.totalNsink<=0
    relativeNsinkstrength=0
   else
    relativeNsinkstrength=pet.DLN/vars.totalNsink
   end
   if pet.DLN>relativeNsinkstrength*(vars.totalN)
    pet.Navaliable=relativeNsinkstrength*(vars.totalN)
   else
    pet.Navaliable=pet.DLN
   end
   pet.LNarea = pet.Navaliable/max(1e-8,pet.area)
end

for int in all_internodes
    vars.totalphotoarea+=int.area
   if vars.totalNsink<=0
    relativeNsinkstrength=0
   else
    relativeNsinkstrength=int.DLN/vars.totalNsink
   end
   if int.DLN>relativeNsinkstrength*(vars.totalN)
    int.Navaliable=relativeNsinkstrength*(vars.totalN)
   
   else
    int.Navaliable=int.DLN
   end
   int.LNarea = int.Navaliable/max(1e-8,int.area)
end

# if vars.totalphotoarea>0
# vars.SLN=vars.totalN/vars.totalphotoarea
# vars.availN=vars.totalN
# else
#     vars.SLN=0.0
# end


end
function reset_Nsink!(vars,all_leaves,all_petiole,all_internodes,all_grains)
 vars.rootNsink=0.0
 for l in all_leaves
    
 end
    for pet in all_petiole
       
    end
    for int in all_internodes
        int.Nsink=0.0
    end
    for g in all_grains
        g.Nsink=0.0
    end
end

function reset_N!(vars,all_leaves,all_petiole,all_internodes)
 vars.totalNsink=0.0
 vars.totalNphotosink=0.0
 vars.nodulebiomass=0.0
 vars.totalphotoarea=0.0
 vars.totalNsinkstruct=0.0
 for l in all_leaves
    l.Navaliable=0.0
 end
    for pet in all_petiole
        pet.Navaliable=0.0
    end
    for int in all_internodes
        int.Navaliable=0.0
    end
end
end