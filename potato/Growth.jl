#=
Version: 1.0.0
Creator: Angela Losu
Creation Date: 2025-12-01
Code description:
This module defines the data structures and parameters specific to the potato species within the Virtual Plant Lab framework.

Credit to Jie Lu for the original wheat model code.
=#

module Growth

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

#Growth
function leaf_dims(biomass, vars, leaf)
    leaf_biomass = biomass
    leaf_area = (leaf_biomass/(vars.SLW))
        for i in 1:100
     leaf.normarea +=(((1-(1-i/100))/(1-vars.max_width))*((1-i/100)/vars.max_width)^(vars.max_width/(1-vars.max_width)))^vars.shapeCoeff*0.01
     end
    leaf_length  = ifelse(leaf.area <= 1e-6, 1e-6,sqrt(leaf_area*vars.LS/leaf.normarea))
    leaf_width   = leaf_length/vars.LS

    return leaf_length, leaf_width, leaf_area
end

function pet_dims(biomass, vars)
    petiole_biomass = biomass
    petiole_length  = vars.specificPetiolelength*petiole_biomass/100
    petiole_width   = petiole_length/40
    petiole_area   = petiole_length * petiole_width
    return petiole_length, petiole_width, petiole_area
end

function int_dims(biomass, vars, int)
    int_biomass = biomass
    int_length  = int_biomass*vars.SIW *SAScoefficient!(vars)
    int_width   = vars.maxwidthint/(1+exp(-0.02*(int.ageD-(0.5*vars.internode_expansion))))
    int_area   =int_length*pi*int_width
    return int_length, int_width,int_area
end


function tuber_dims(biomass, vars, tuber)
    tuber_biomass = biomass
    tuber_length  = 0.05
    tuber_width   = 0.01
    return tuber_length, tuber_width
end

function rootmer_dims(biomass,vars,rootmer)
    rootmer_biomass=biomass
    rootmer_volume=rootmer_biomass/(vars.RTD*1e3)
    rootmer_length=rootmer_volume/(pi*(vars.width/2)^2)+rootmer.difflength
    rootmer_width=vars.width
    rootmer_rootseg= Int(trunc(rootmer_length/vars.IBD))  
    rootmer_actlength=rootmer_rootseg*vars.IBD
    rootmer.difflength=max(0,rootmer_length-rootmer_actlength)
return rootmer_actlength,rootmer_width,rootmer_rootseg
end


function lrootmer_dims(biomass,vars,lrootmer)
    lrootmer_biomass=biomass
    lrootmer_volume=lrootmer_biomass/(vars.RTD*1e3)
    lrootmer_width=vars.width*vars.RDM
    lrootmer_length=lrootmer_volume/(pi*(lrootmer_width/2)^2)+lrootmer.difflength
    lrootmer_rootseg= Int(trunc(lrootmer_length/vars.IBD))
    lrootmer_val = lrootmer.val+1
    lrootmer_actlength=lrootmer_rootseg*vars.IBD
    lrootmer.difflength=max(0,lrootmer_length-lrootmer_actlength)
return lrootmer_actlength,lrootmer_width,lrootmer_rootseg,lrootmer_val
end


function root_dims(biomass, vars,root)
if root.order==0
    root_biomass=max(0,biomass+root.biomass)
    root_volume=root_biomass/(vars.RTD*1e3)
    root_length=vars.IBD
    if vars.monocot==false
    root_width=2*sqrt(root_volume/(pi*root_length))
    else
    root_width=vars.width
    end
elseif root.order==1
     root_biomass=biomass*vars.RDM^2
     root_length=vars.IBD
     root_width=vars.width*vars.RDM
end
    return root_length, root_width
    end


#calculate the sinks
function sink(te,wmax,rg,Age,Agepre)
    tm=te/2
    maxsinkstrength=wmax*((2*te-tm)/(te*(te-tm)))*(tm/te)^(tm/(te-tm))
    result=maxsinkstrength/(1-rg) * ((te-Age)/(te-tm))*(Age/tm)^(tm/(te-tm))
    if(Age>te)
        return 0
    else
    return result*(Age-Agepre)
     end
end

#for root system
function rootsegsink(root,vars,width)
    #println("total auxin$(root.totalauxin)")
    if vars.ELauxin==true
    root.ELN=vars.EL*(1.746*exp(-0.343*root.totalauxin*root.coef)+1)
    else
        root.ELN=vars.EL
    end
    if root.age<50
    potsink=max(0,root.ELN*width*pi*(width/2)^2*vars.RTD*1e3)
    else
    potsink=0
    end
return potsink
end

function radialrootsegsink(root,vars)
    if vars.monocot==false&& root.order==0
    potsink=(root.totalarea)*vars.IBD*vars.RTD*1e3
    else
    potsink=0    
    end
    return potsink
    end

    function finerootsink(length, vars)
        totalfinerootsink::Float64=0.0
        finerootsink=vars.finerootD*vars.EL*pi*(vars.finerootD/2)^2*vars.RTD*1e3
    maxfinerootsink= vars.finerootdensity*length*pi*(vars.finerootD/2)^2*vars.RTD*1e3
    totalfinerootsink+=finerootsink
    return finerootsink, maxfinerootsink, totalfinerootsink
    end
    
    function nodulesink(root,vars)
    if root.Nodule==true&&vars.species=="faba"
        nodulesink=root.biomass*0.04
    else
        nodulesink=0.0
    end
    return nodulesink
    end

sink_strengthroot(vars) =sink(vars.root_expansion,vars.root_potentialbiom,vars.rg,vars.root_ageD,vars.root_ageDprevious)*RLratio(vars)

sink_strengtharoot(rmer,vars)=rootsegsink(rmer,vars,vars.width)
sink_strengthlroot(lrmer,vars)=rootsegsink(lrmer,vars,vars.width*vars.RDM)
sink_strengthfineroot(length,vars)=finerootsink(length,vars)

#calculate the RL ratio for plasticity
function RLratio(vars)
 if vars.RLplasticity==true
    if vars.Nc<0.01
        RLratio=2.747*exp(-99.974*0.01)+1
    else
        RLratio=2.747*exp(-99.974*vars.Nc)+1
    end
else
    RLratio=1
end 
    return RLratio
end


function age!(all_leaves, all_internodes,all_tubers,all_petiole, all_meristems,all_rootmeristem, all_lrootmeristem,all_rootBudNode,all_shootrootNode,vars,dayofyear)
    
    for leaf in all_leaves
        leaf.agewait+=max(0,Main.temperature(dayofyear)-vars.tb)
        ##leaf emerged after internode emerge
        if leaf.agewait>=(vars.delay)
        leaf.age += 1
        leaf.ageDprevious=leaf.ageD
        leaf.ageD +=max(0,Main.temperature(dayofyear)-vars.tb)
        
        end
    end
    for int in all_internodes
        int.agewait+=max(0,Main.temperature(dayofyear)-vars.tb)
        if int.agewait>=(vars.delay)
        int.age += 1
        int.ageDprevious=int.ageD
        int.ageD +=max(0,Main.temperature(dayofyear)-vars.tb)
        end
    end

      
    for g in all_tubers
        g.age += 1
        g.ageDprevious=g.ageD
        g.ageD +=max(0,Main.temperature(dayofyear)-vars.tb)
    end

    for pet in all_petiole
        pet.agewait+=max(0,Main.temperature(dayofyear)-vars.tb)
        if pet.agewait>(vars.delay)
        pet.age += 1
        pet.ageDprevious=pet.ageD
        pet.ageD += max(0,Main.temperature(dayofyear)-vars.tb)
        end
    end
    
    for mer in all_meristems
        mer.age += 1
        mer.ageDprevious=mer.ageD
        mer.ageD += max(0,Main.temperature(dayofyear)-vars.tb)
        
    end

    for amer in all_rootmeristem
        amer.age += 1
        amer.ageDprevious=amer.ageD
        amer.ageD += max(0,Main.temperature(dayofyear)-vars.tb)
    end

    for lmer in all_lrootmeristem
        lmer.age += 1
        lmer.ageDprevious=lmer.ageD
        lmer.ageD += max(0,Main.temperature(dayofyear)-vars.tb)
    end

    for rb in all_rootBudNode
        rb.age += 1
        rb.ageDprevious=rb.ageD
        rb.ageD += max(0,Main.temperature(dayofyear)-vars.tb)
        
    end
    ##shoot root node age calculated,related to calculate axial root branching
    for srb in all_shootrootNode
        srb.age += 1
        srb.ageD += max(0,Main.temperature(dayofyear)-vars.tb)
        if srb.rank==0&&srb.age<8
            srb.er+=0
        else
        srb.er+= Main.temperature(dayofyear)*vars.ER
        end
    end
      #root system age increase
     vars.root_age+=1
     vars.root_ageDprevious=vars.root_ageD
     vars.root_ageD+= max(0,Main.temperature(dayofyear)-vars.tb)
     #plant age calculated
     vars.age+=1
     vars.ageDprevious=vars.root_ageD
     vars.ageD+= max(0,Main.temperature(dayofyear)-vars.tb)
    return nothing
end


#   function photosynthesis!(all_leaves,all_petiole,all_internodes,vars,DL)
#   #total assimilated carbon umol/day
#     assimilate=0.0
#       for l in all_leaves
#       if vars.Nphoto==true
#      if l.LNarea>0.18
#           l.Amax=vars.lambda*(2/(1+exp(-12.25*(l.LNarea-0.18)))-1)
#      else
#          l.Amax=0
#      end
#   else
#       l.Amax=vars.lambda
#   end
#  println("Amax $(l.Amax) LNarea $(l.LNarea) fabs $(l.fabs) PAR $(l.PAR)")
#  if l.Amax>0
#          assimilate+=l.Amax*(1-exp(-vars.alpha*l.PAR/(l.Amax)))*l.area*DL
#      else
#           assimilate+=0
#               end
#    end

#   for pet in all_petiole
#      if vars.Nphoto==true
#      if pet.LNarea>0.18
#          pet.Amax=vars.lambda*(2/(1+exp(-12.25*(pet.LNarea-0.18)))-1)
#      else
#          pet.Amax=0
#      end
#   else
#       pet.Amax=vars.lambda
#  end

#  if pet.Amax>0
#     assimilate+=pet.Amax*(1-exp(-vars.alpha*pet.PAR/(pet.Amax)))*pet.area*DL
#   else
#       assimilate+=0
#   end
#  end    


#      for int in all_internodes
#      if vars.Nphoto==true
#      if int.LNarea>0.18
#           int.Amax=vars.lambda*(2/(1+exp(-12.25*(int.LNarea-0.18)))-1)
#      else
#           int.Amax=0
#     end
#  else
#      int.Amax=vars.lambda
#   end

#  if int.Amax>0    
#   assimilate+=int.Amax*(1-exp(-vars.alpha*int.PAR/(int.Amax)))*int.area*DL
#   else
#      assimilate+=0
#   end
#   end
#   return assimilate
#   end

# function NrelatedRUE!(vars)
# if vars.SLN>0.3
#    #vars.Amax=10.41*vars.SLN+10.59 #Wang dong 2024
#     vars.Amax=34.1*(2/(1+exp(-1.4*(vars.SLN-0.3)))-1) #Sinclair and Horie,1989 
# else
#     vars.Amax=0.0
# end

# if vars.SLN>0.3   
# #vars.RUE=5.67*(1-exp(-(vars.Nc*100-0.3))/1.6) #Comparative response of wheat and oilseed rape to nitrogen supply: absorption and utilisation efficiency of radiation and nitrogen during the reproductive stages determining yield
# vars.RUE=1.3*(2-1.83*exp(-0.9*vars.Amax*44.01/1000))
# else
#     vars.RUE=0.0
# end
# return nothing
# end

function SAScoefficient!(vars)
max=2
min=1
k=5
if Main.SAS==true
return (max-min)*exp(-k*vars.sr)+min
else
    return 1
end
end


function grow!(DL,species, all_leaves, all_internodes, all_tubers, all_petiole,all_rootmeristem,all_lrootmeristem,all_roots,all_aroots,all_rootbud,all_meristems,all_budnodes,all_emgrootbud)
    # Compute total biomass increment
    mvars = data(species)
    rootbiomass=0.0+mvars.leftrootbiomass
    # Total sink strength
    root_sink=0.0
    aroot_sink=0.0
    mCO2=44.01#CO2 molar mass(gr/mol)
    rm=0.015#maintance respiration
    fCO2=0.71 #CONVERSION FACTOR from gr assimilates to gr biomass 
    mvars.previousbiomass=mvars.biomass
    ΔB::Float64=0
    growth::Float64=0
   #radial growth 
   totalarea=zeros(length(all_aroots))
   # Calculate the total area of all the lateral roots
     for erb in all_emgrootbud
         val=erb.val+1
         val>1&&(totalarea[1:val-1].+=pi*(mvars.width*mvars.RDM/2)^2)
      end
    
     
    mvars.total_sink=0.0
    for leaf in all_leaves
       #total sink strength of a plant
        mvars.total_sink += sink(mvars.leaf_expansion,mvars.potentialbiom_leaf,mvars.rg,leaf.ageD,leaf.ageDprevious)*0.95
        #total sink strength per leaf
        leaf.total_sink=sink(mvars.leaf_expansion,mvars.potentialbiom_leaf,mvars.rg,leaf.ageD,leaf.ageDprevious)*0.95
        #total sink strength of a plant per day
        mvars.total_sinkperday+= sink(mvars.leaf_expansion,mvars.potentialbiom_leaf,mvars.rg,leaf.ageD,leaf.ageDprevious)*0.95
        #leaf produced auxin available for root system
        if leaf.age<7&&leaf.order<5
        mvars.auxp+=leaf.auxp
        end
    end
     #println("auxin $(mvars.auxp)")
    #when no leaves are present, auxin is produced by the meristem
    if mvars.leafnumber<1&&mvars.age<20
        mvars.auxp =1
    end

    for pet in all_petiole
       #total sink strength of a plant
        mvars.total_sink += sink(mvars.leaf_expansion,mvars.potentialbiom_leaf,mvars.rg,pet.ageD,pet.ageDprevious)*0.05
        #total sink strength per petiole
        pet.total_sink=sink(mvars.leaf_expansion,mvars.potentialbiom_leaf,mvars.rg,pet.ageD,pet.ageDprevious)*0.05
        #total sink strength of a plant per day
        mvars.total_sinkperday+= sink(mvars.leaf_expansion,mvars.potentialbiom_leaf,mvars.rg,pet.ageD,pet.ageDprevious)*0.05
    end
    
    for int in all_internodes
        #regular internode sink strength 
        if int.order>mvars.max_shortint
        #total sink strength of a plant
        mvars.total_sink += sink(mvars.internode_expansion,mvars.potentialbiom_internode,mvars.rg,int.ageD,int.ageDprevious)
        #total sink strength per internode
        int.total_sink =sink(mvars.internode_expansion,mvars.potentialbiom_internode,mvars.rg,int.ageD,int.ageDprevious)
        #total sink strength of a plant per day
        mvars.total_sinkperday+=sink(mvars.internode_expansion,mvars.potentialbiom_internode,mvars.rg,int.ageD,int.ageDprevious)
        else
            #when the internode is a short internode, it does not contribute to the sink strength
            mvars.total_sink+=0
            int.total_sink=0
            mvars.total_sinkperday+=0
        end
    end

    
    for g in all_tubers
        #total sink strength of a plant
        mvars.total_sink += sink(mvars.te_tuber,mvars.potentialbiom_tuber,mvars.rg,g.ageD,g.ageDprevious)
        #total sink strength per tuber
        g.total_sink=sink(mvars.te_tuber,mvars.potentialbiom_tuber,mvars.rg,g.ageD,g.ageDprevious)
        #total sink strength of a plant per day
        mvars.total_sinkperday+=sink(mvars.te_tuber,mvars.potentialbiom_tuber,mvars.rg,g.ageD,g.ageDprevious)
        
    end
    #root system sink strength (where the root system is considered as a single organ)
    if mvars.root_age<20&&mvars.actbiomass_leaf<=0
        mvars.total_sink+=0.001
        mvars.total_sinkperday+=0.001
        mvars.total_rootsink+=0.001
        mvars.root_sinkperday=0.001
    else  
    #total sink strength of a plant    
        mvars.total_sink+=sink_strengthroot(mvars)
    #total sink strength of a plant per day
        mvars.total_sinkperday+=sink_strengthroot(mvars)
        #total sink strength per root system
        mvars.total_rootsink+=sink_strengthroot(mvars)
        #total sink strength per root system per day
        mvars.root_sinkperday=sink_strengthroot(mvars)
    end
    #calculate the per root meristem sink strength (where the root system is conbination of all root segments)    
    #one root has one root meristem
    for rmer in all_rootmeristem
        #total root sink strength of the root system per day 
        root_sink+=sink_strengtharoot(rmer,mvars)
        # total axial root sink strength per root meristem per day
        rmer.aroot_sinkperday=sink_strengtharoot(rmer,mvars)
       #total axial root sink strength 
        mvars.aroot_sink+=sink_strengtharoot(rmer,mvars)
        #total axial root sink strength per root meristem
        rmer.total_sink+=sink_strengtharoot(rmer,mvars)
        
    end
    
           #calculate the radial root sink strength for axial root (legume roots and nodules)    
            for r in all_roots
            #calculate the area of all the lateral root happened after the formation of the axial root  
            r.totalarea=totalarea[r.val]
            #total root sink strength of the root system per day
            root_sink+=radialrootsegsink(r,mvars)
            # total axial root sink strength of the root system 
            mvars.aroot_sink+=radialrootsegsink(r,mvars)
            #total axial root sink strength per axial root per day
            r.total_sink=radialrootsegsink(r,mvars)
            ##total root sink strength of the root system per day
            root_sink+=nodulesink(r,mvars)
            #total nodule sink strength of the root system
            mvars.nodule_sink+=nodulesink(r,mvars)
            #total nodule sink strength per axial root per day
            r.nodulesink=nodulesink(r,mvars)
            if r.order==1
            if sink_strengthfineroot(mvars.IBD,mvars)[3] < sink_strengthfineroot(mvars.IBD,mvars)[2]
                 root_sink+=sink_strengthfineroot(mvars.IBD,mvars)[1]
                 r.finerootsink+=sink_strengthfineroot(mvars.IBD,mvars)[1]
                 r.fineroot_sinkperday=sink_strengthfineroot(mvars.IBD,mvars)[1]
            end
           end
        end

    for lrmer in all_lrootmeristem
         #total root sink strength of the root system per day (lateral root )
        root_sink+=sink_strengthlroot(lrmer,mvars)*5
        # total lateral root sink strength per lateral root meristem per day
        lrmer.lroot_sinkperday =sink_strengthlroot(lrmer,mvars)*5
        #total lateral root sink strength per lateral root meristem
        lrmer.total_sink+=sink_strengthlroot(lrmer,mvars)*5
        end
        
        #assimilates can be allocated to the plant (photosynthesis or fixed ratio of sink strength)

        if Main.raytrace==true
            Cpoolprevious=mvars.Cpool
            ΔB =max(0,((mvars.Ag/1e6*mCO2-mvars.previousbiomass*rm)*fCO2-mvars.carcoeff*mvars.Nfix+Cpoolprevious)) 
            #max(0,max(0,mvars.RUE*mvars.PAR/(1e6)-mvars.carcoeff*mvars.Nfix+mvars.Cpool))
            if mvars.Cpool>0
            mvars.Cpool=ΔB-mvars.total_sinkperday
            else
            mvars.Cpool=0
            end
            growth=ΔB
            #mvars.biomass+=growth
           else
            if mvars.RLplasticity==true
            ΔB=max(0,0.7*sink(1500,20,0.3,mvars.ageD,mvars.ageDprevious)-mvars.carcoeff*mvars.Nfix) #assimilates is a fraction of whole plant sink strength
            growth=ΔB
            else
            ΔB =max(0,0.7*mvars.total_sinkperday-mvars.carcoeff*mvars.Nfix)# assimilates is a fraction of total sink strength of a day
            growth=ΔB 
           end
           end
            println("Ag$(mvars.Ag) mvars.total_sinkperday$(mvars.total_sinkperday) mvars.Cpool$(mvars.Cpool) ΔB$(ΔB)")
           #sink sources ratio per day, and calculate the dominance of the tiller
           mvars.sr=ΔB/(mvars.total_sinkperday)
            for bn in all_budnodes
            bn.dom= 1/(0.5*mvars.sr)+0.5
            end
        
         for mer in all_meristems
             mer.dom=1/(0.5*mvars.sr)+0.5
        end
           

    # Allocate biomass to leaves and internodes
    for leaf in all_leaves
        #println("leafbiomass$(leaf.biomass) leafsink$(leaf.total_sink) ")
        if leaf.ageD<mvars.leaf_expansion
         leaf.biomassprevious=leaf.biomass
        leaf.biomass += min(leaf.total_sink,growth*leaf.total_sink/mvars.total_sinkperday)
        mvars.actbiomass+=leaf.biomass
        mvars.actbiomass_leaf+=leaf.biomass
        leaf.Δbiomass=leaf.biomass-leaf.biomassprevious
        else
            leaf.biomass+=0
            mvars.actbiomass+=leaf.biomass
        mvars.actbiomass_leaf+=leaf.biomass
        end
        #println("leafbiomass$(leaf.biomass) leafsink$(leaf.total_sink)area$(leaf.area)")
        end

        for pet in all_petiole
            if pet.ageD<mvars.leaf_expansion
                pet.biomassprevious=pet.biomass
              pet.biomass +=min(pet.total_sink,growth*pet.total_sink/mvars.total_sinkperday)
              
              mvars.actbiomass+=pet.biomass
              mvars.actbiomass_leaf+=pet.biomass
                pet.Δbiomass=pet.biomass-pet.biomassprevious
             
            else
                pet.biomass+=0
                 mvars.actbiomass+=pet.biomass
              mvars.actbiomass_leaf+=pet.biomass
            end
    end
 

  for int in all_internodes
    if int.ageD<mvars.internode_expansion
        int.biomassprevious=int.biomass
    int.biomass += min(int.total_sink,growth*int.total_sink/mvars.total_sinkperday)
    #mvars.Cpool-=min(int.total_sink,growth*int.total_sink/mvars.total_sinkperday)
    mvars.actbiomass+= int.biomass
    mvars.actbiomass_stem+=int.biomass
    int.Δbiomass=int.biomass-int.biomassprevious
   
    else
        int.biomass+=0
        mvars.actbiomass+= int.biomass
    mvars.actbiomass_stem+=int.biomass
    end
      #println("stembiom $(int.total_sink)")
 end

  
     for g in all_tubers
        g.biomassprevious=g.biomass
    g.biomass+=min(g.total_sink,growth*g.total_sink/mvars.total_sinkperday)
    #mvars.Cpool-=min(g.total_sink,growth*g.total_sink/mvars.total_sinkperday)
    mvars.actbiomass+=g.biomass 
    mvars.actbiomass_tuber+=g.biomass
    g.Δbiomass=g.biomass-g.biomassprevious
    
   
    end
        #root biomass allocated to axial and lateral roots
            if mvars.root_ageD<mvars.root_expansion
                #mvars.Cpool-=min(sink_strengthroot(mvars),growth*sink_strengthroot(mvars)/mvars.total_sinkperday)
                
            mvars.root_biomass+=min(sink_strengthroot(mvars),growth*sink_strengthroot(mvars)/mvars.total_sinkperday)
            rootbiomass+=min(sink_strengthroot(mvars),growth*sink_strengthroot(mvars)/mvars.total_sinkperday)
                else
                   mvars.root_biomass+=0
                    rootbiomass+=0
                end          
       
      
        
        for rmer in all_rootmeristem
            rmer.biomassprevious=rmer.biomass
            rmer.biomass +=min(sink_strengtharoot(rmer,mvars),rootbiomass*rmer.aroot_sinkperday/root_sink)
            
            mvars.actbiomass_rootperday+=min(rmer.aroot_sinkperday,rootbiomass*rmer.aroot_sinkperday/root_sink) 
            mvars.actbiomass_root+=rmer.biomass   
            mvars.actbiomass+=rmer.biomass
           
            rmer.Δarootbiom=rmer.biomass-rmer.biomassprevious
            #shoot auxin
            if mvars.Naux==true
            rmer.fracshootaux=mvars.auxp*(rmer.aroot_sinkperday/root_sink)*max(0.226,min(1,1.2-20.37*mvars.Nc))
            else
            rmer.fracshootaux=mvars.auxp*(rmer.aroot_sinkperday/root_sink)*0.226   
            end
            rmer.totalmerauxin+=rmer.merauxinN
            rmer.totalauxin +=(rmer.fracshootaux+rmer.merauxinN)
            rmer.totalauxin=(1-rmer.decay)*rmer.totalauxin
            rmer.shootauxin+=mvars.auxp*(rmer.aroot_sinkperday/root_sink)
            rmer.totalauxinperday=rmer.fracshootaux+rmer.merauxinN
            #println("shootauxin$(rmer.shootauxin)")
            #println("totalauxin $(rmer.totalauxin)")
            #println("merauxin $(rmer.merauxinN)")
        end

        for r in all_roots
            if r.order==0
                r.biomass=mvars.IBD*(mvars.width/2)^2*pi*mvars.RTD*1e3
            else
                r.biomass=mvars.IBD*(mvars.width*mvars.RDM/2)^2*pi*mvars.RTD*1e3
            end
            r.biomassprevious=r.biomass
            r.biomass+=min(radialrootsegsink(r,mvars),rootbiomass*r.total_sink/root_sink)
            r.biomassradi+=min(radialrootsegsink(r,mvars),rootbiomass*r.total_sink/root_sink)
            mvars.actbiomass_root+=min(radialrootsegsink(r,mvars),rootbiomass*r.total_sink/root_sink)
            mvars.actbiomass_rootperday+=min(radialrootsegsink(r,mvars),rootbiomass*r.total_sink/root_sink)
            mvars.actbiomass+=r.biomassradi
           ##if nodules are present, allocate biomass to nodules
            r.nodulebiomass=min(nodulesink(r,mvars),rootbiomass*r.nodulesink/root_sink)
            if r.order==1
            r.finerootbiomass+=min(sink_strengthfineroot(mvars.IBD,mvars)[1], rootbiomass*r.fineroot_sinkperday/root_sink)
                     mvars.actbiomass_root+=min(r.fineroot_sinkperday, rootbiomass*r.fineroot_sinkperday/root_sink)
                    mvars.actbiomass+=min(r.fineroot_sinkperday, rootbiomass*r.fineroot_sinkperday/root_sink)
                    mvars.actbiomass_rootperday+=min(r.fineroot_sinkperday, rootbiomass*r.fineroot_sinkperday/root_sink) 
                    r.nodulebiomass=min(nodulesink(r,mvars),rootbiomass*r.nodulesink/root_sink)
            end
            
            end 
 
            for lrmer in all_lrootmeristem
                #println("$(lrmer.rank)calc $(lrmer.rootseg)")
                lrmer.biomassprevious=lrmer.biomass
                lrmer.biomass +=min(sink_strengthlroot(lrmer,mvars), rootbiomass*lrmer.lroot_sinkperday/root_sink)
                
                lrmer.Δlrootbiom=lrmer.biomass-lrmer.biomassprevious
                
                mvars.actbiomass_root+=lrmer.biomass
                 
                mvars.actbiomass_rootperday+=min(lrmer.lroot_sinkperday, rootbiomass*lrmer.lroot_sinkperday/root_sink)
                mvars.actbiomass+=lrmer.biomass
                #####shoot auxin production and allocation to lateral root meristem
                lrmer.shootauxin+=mvars.auxp*(lrmer.lroot_sinkperday/root_sink)*max(0.226,min(1,1.2-20.37*mvars.Nc))
                lrmer.Δshootauxin =mvars.auxp*(lrmer.lroot_sinkperday/root_sink)*max(0.226,min(1,1.2-20.37*mvars.Nc))
                lrmer.totalmerauxin+=lrmer.merauxinN
                lrmer.totalauxin +=(lrmer.Δshootauxin+lrmer.merauxinN) #*(mvars.Nc/(0.025+mvars.Nc))
                lrmer.totalauxin=(1-lrmer.decay)*lrmer.totalauxin
            #     println("lrmer$(lrmer.merauxinN)")
                #println("$(lrmer.rank)totalauxin$(lrmer.ELN)")
            #     println("shootauxin$(lrmer.shootauxin)")       
             end
            
            #if root segments sink strength is than the total root sink strength, then left root biomass will be included for the next day
            mvars.leftrootbiomass=max(0,rootbiomass-mvars.actbiomass_rootperday)
          mvars.rs=mvars.actbiomass_root/(mvars.actbiomass_tuber+mvars.actbiomass_stem+mvars.actbiomass_leaf)
          println("actrootbiom$(mvars.actbiomass_root)")
    return nothing

end

#reset
function reset_sink!(all_leaves,all_internodes,all_tubers,all_petiole,all_rootmeristem,all_lrootmeristem,vars)
    vars.total_sinkperday=0.0
    vars.auxp=0.0
    vars.potNup=0.0
    vars.Ag=0.0
    for l in all_leaves
    l.total_sink=0.0
    l.normarea=0
    end
    for int in all_internodes
        int.total_sink =0.0
    end
    
    for g in all_tubers
        g.total_sink=0.0
    end
    for pet in all_petiole
        pet.total_sink=0.0
    end
    for rmer in all_rootmeristem
        rmer.total_sink=0.0
        
    end
    for lrmer in all_lrootmeristem
        lrmer.total_sink=0.0
        
    end
    return nothing
    end
function reset_biomass!(field)
    @threads for species in field
        data(species).actbiomass=0.0
        data(species).actbiomass_leaf=0.0
        data(species).actbiomass_stem=0.0
        data(species).actbiomass_tuber=0.0
        data(species).actbiomass_root=0.0
        data(species).actbiomass_rootperday=0.0
    end
end

function reset_auxp!(field)
   @threads for species in field
        data(species).auxp = 0.0
    end
    return nothing
end

function size_leaves!(all_leaves, vars)
    for leaf in all_leaves
        leaf.length, leaf.width, leaf.area = leaf_dims(leaf.biomass, vars, leaf)
        
    end
    return nothing
end

function size_internodes!(all_internodes, vars)
    for int in all_internodes
        int.length, int.width, int.area = int_dims(int.biomass, vars,int)
    end
    return nothing
end


function size_tubers!(all_tubers, vars)
    for g in all_tubers
        g.length, g.width = tuber_dims(g.biomass, vars,g)
    end
    return nothing
end

function size_petiole!(all_petiole, vars)
    for pet in all_petiole
        pet.length, pet.width, pet.area = pet_dims(pet.biomass, vars)
    end
    return nothing
end

function is_linked(lrn, lrmer)
    # Example: Check if the positions of `lrootNode` and `lrootmeristem` are close
    return abs(lrn.x - lrmer.x) < 1e-4&& abs(lrn.y - lrmer.y) < 1e-4 && abs(lrn.z - lrmer.z) < 1e-4
end

function size_rootmer!(all_rootmeristem,all_rootNodes, vars)
    for rmer in all_rootmeristem
        rmer.length, rmer.width, rmer.rootseg = rootmer_dims(rmer.Δarootbiom, vars,rmer)
        rmer.totallength+=rmer.length
        vars.rootlength+=rmer.length
        vars.arootlength+=rmer.length
        #rmer.colors=RGB(rmer.totalauxin/10,0,1-rmer.totalauxin/10)
        
        for rn in all_rootNodes
            if is_linked(rn,rmer)
            rn.rootseg=rootmer_dims(rmer.Δarootbiom, vars,rmer)[3]
                rn.totalauxin=rmer.totalauxin
                rn.fracshootaux=rmer.fracshootaux
            end
        end
    end

    return nothing

end


function size_lrootmer!(all_lrootmeristem,all_lrootNodes, vars)
   
    for lrmer in all_lrootmeristem
        lrmer.length, lrmer.width, lrmer.rootseg,lrmer.val = lrootmer_dims(lrmer.Δlrootbiom, vars,lrmer)
        lrmer.totallength+=lrmer.length
        vars.rootlength+=lrmer.length
        vars.lrootlength+=lrmer.length
        for lrn in all_lrootNodes
            if is_linked(lrn,lrmer)
            lrn.rootseg=lrmer.rootseg
            break
            end
        end
    end
    return nothing
end

function size_root!(all_root,vars)
    for r in all_root
    r.length, r.width = root_dims(vars.RTD*pi*(vars.width/2)^2*vars.IBD*1e3,vars,r)
    end
    return nothing
    end



#function calculate_PAR!(field,all_leaves,all_petiole,all_internodes,vars,DOY)
   
   # Run the ray tracer to compute daily PAR absorption
   #Main.run_raytracer!(field; DOY)
   #ref= Main.run_raytracer!(field; DOY)[2]*Main.DL/(Main.field_length* Main.field_width) #reference PAR absorbed by a leaf with 1 m^2 area
   #Main.canopy_photosynthesis!(field)
   

   # Add up PAR absorbed by each leaf within each plant
    #    for l in all_leaves
    #        l.absm= l.PAR
    #        vars.PAR += power(l.material)[1]
    #        l.fabs = l.absm/Main.ref
    #    end
    #    for pet in all_petiole
    #        pet.PAR = power(pet.material)[1]
    #        pet.absm= pet.PAR
    #        vars.PAR += power(pet.material)[1]
    #        pet.fabs = pet.absm/Main.ref
        
    #    end

    #    for int in all_internodes
    #        int.PAR = power(int.material)[1]
    #        int.absm= int.PAR
    #        vars.PAR += power(int.material)[1]
    #        int.fabs = int.absm/Main.ref
           
    #    end
       #println("Total PAR absorbed by the plant: $(vars.PAR)")
   #return nothing
#end



function calculate_diffuse!(;scene,acc_scene,settings,field,lat = 52.0*π/180.0, DOY = sowingdate)
    dome = sky(acc_scene,
                  Idir = 0.0, # No direct solar radiation
                  Idif = 1.0, # In order to get relative values
                  nrays_dif = 1_000_000, # Total number of rays for diffuse solar radiation
                  sky_model = StandardSky, # Angular distribution of solar radiation
                  dome_method = equal_solid_angles, # Discretization of the sky dome
                  ntheta = 9, # Number of discretization steps in the zenith angle
                  nphi = 12) # Number of discretization steps in the azimuth angle
#settings = RTSettings(pkill = 0.9, maxiter = 4, nx = 5, ny = 5,dx=10,dy=10,parallel = true)
rtobj= RayTracer(acc_scene, materials(scene), dome, settings);
trace!(rtobj)
@threads for sp in field
    for leaf in apply(sp,Query(Main.Plant.Leaf))
        leaf.PARdif=power(leaf.material)[1]/leaf.area
    end

     for int in apply(sp,Query(Main.Plant.Internode))
        int.PARdif=power(int.material)[1]/int.area
    end

    for pet in apply(sp,Query(Main.Plant.Petiole))
        pet.PARdif=power(pet.material)[1]/pet.area
    end
end
                end

ref::Float64=0.0
    function calculate_photosynthesis!(;scene, acc_scene,settings, field, lat = 52.0*π/180.0,DOY=sowingdate,
                 f = 0.5, w = 0.5,DL=12*3600)
    # Compute the solar irradiance assuming clear sky conditions
    Ig, Idir, Idif = clear_sky(lat = lat, DOY = DOY, f = f)
    # Conversion factors to PAR for direct and diffuse irradiance
    PARdir = Idir*waveband_conversion(Itype = :direct,  waveband = :PAR, mode = :flux)
    PARdif = Idif*waveband_conversion(Itype = :diffuse, waveband = :PAR, mode = :flux)
    Itot = (mean(PARdir) + mean(PARdif))*w
    ref = Itot  # Store the value for later use
    # Create the light source for the ray tracer
    dome = sky(acc_scene, Idir = PARdir, nrays_dir = 100_000, 
             Idif = 0, # Diffuse solar radiation from above
             )
             
    # Ray trace the scene
    rtobj = RayTracer(acc_scene, materials(scene), dome, settings)
    trace!(rtobj)
    # Transfer power to PARdif
    @threads for sp in field
        for l in apply(sp,Query(Main.Plant.Leaf))
             l.PARdir = power(l.material)[1]/l.area
             PARdifday = PARdif*l.PARdif
             l.PAR = l.PARdir + PARdifday
             l.PARperday+=l.PAR*DL*w
             #println("l.PAR$(l.PAR)PARdif$(PARdifday) PARdir$(PARdir)")
            #l.PAR=power(l.material)[1]/l.area
            
            data(sp).PAR+=l.PARperday

    if data(sp).Nphoto==true
      if l.LNarea>0.3
           l.Amax=data(sp).lambda*(2/(1+exp(-12.25*(l.LNarea-0.18)))-1)
      else
          l.Amax=0
      end
   else
       l.Amax=data(sp).lambda
   end
  
  if l.Amax>0
          l.Ag+=max(0,(l.Amax*(1-exp(-data(sp).alpha*l.PAR/(l.Amax))))*w*DL)
      else
           l.Ag+=0
              end
        end

        for int in apply(sp,Query(Main.Plant.Internode))
             int.PARdir = power(int.material)[1]/int.area
             PARdifday = PARdif*int.PARdif
             int.PAR = int.PARdir + PARdifday
              int.PARperday+=int.PAR*DL*w
            data(sp).PAR+=int.PARperday
      
       if data(sp).Nphoto==true
        if int.LNarea>0.3
           int.Amax=data(sp).lambda*(2/(1+exp(-12.25*(int.LNarea-0.18)))-1)
        else
           int.Amax=0
        end
      else
           int.Amax=data(sp).lambda
      end

       if int.Amax>0    
       int.Ag+=max(0,(int.Amax*(1-exp(-data(sp).alpha*int.PAR/(int.Amax))))*w*DL)
      else
       int.Ag+=0
      end
        end

    for pet in apply(sp,Query(Main.Plant.Petiole))
             pet.PARdir = power(pet.material)[1]/pet.area
             PARdifday = PARdif*pet.PARdif
             pet.PAR = pet.PARdir + PARdifday
                pet.PARperday+=pet.PAR*DL*w
            #pet.PAR=power(pet.material)[1]/pet.area
            data(sp).PAR+=pet.PARperday

    if data(sp).Nphoto==true
     if pet.LNarea>0.3
         pet.Amax=data(sp).lambda*(2/(1+exp(-12.25*(pet.LNarea-0.18)))-1)
      else
         pet.Amax=0
     end
   else
       pet.Amax=data(sp).lambda
  end

  if pet.Amax>0
     pet.Ag+=max(0,(pet.Amax*(1-exp(-data(sp).alpha*pet.PAR/(pet.Amax))))*w*DL)
   else
      pet.Ag+=0
   end
        end
    end
    return ref
end



function reset_photosynthesis!(field)
    @threads for sp in field
        data(sp).PAR =0.0
        for leaf in apply(sp,Query(Main.Plant.Leaf))
            leaf.Ag = 0.0
            leaf.PAR = 0.0
            leaf.PARperday=0.0
        end

        for int in apply(sp,Query(Main.Plant.Internode))
            int.Ag = 0.0
            int.PAR = 0.0
            int.PARperday=0.0
        end

         for pet in apply(sp,Query(Main.Plant.Petiole))
            pet.Ag = 0.0
            pet.PAR = 0.0
            pet.PARperday=0.0
        end
    end
    return nothing
end

function daily_photosynthesis(field,DOY ; lat = 52.0*π/180.0)
    # Compute fraction of diffuse irradiance per leaf
    acc_scene,settings,scene=Main.create_scene(field)
    calculate_diffuse!(scene = scene, acc_scene = acc_scene,settings=settings ,field = field,
                        DOY = DOY, lat = lat);
    # Gaussian quadrature over the
    NG = 5
    f, w = gausslegendre(NG)
    w ./= 2.0
    f .= (f .+ 1.0)/2.0
    # Reset PARfield,all_leaves,all_petiole,all_internodes;
     reset_photosynthesis!(field)
    # Loop over the day
    dec = declination(DOY)
    DL = day_length(lat, dec)*3600
    r=Float64[0,0,0,0,0]
    for i in 1:NG
        #println("step $i out of $NG")
       r[i]=calculate_photosynthesis!(scene = scene, acc_scene = acc_scene,settings=settings, field = field,
                                  f = f[i], w = w[i], DL = DL, DOY = DOY, lat = lat)

                                end
                                ref= sum(r)*DL   
                return ref        
end

 function canopy_photosynthesis!(field,DOY)
     # Integrate photosynthesis over the day at the leaf level
     ref=daily_photosynthesis(field,DOY)/(Main.field_length*Main.field_width)#reference PAR with 1 m^2 area
     println("ref $ref")
     # Aggregate to the  level
     #Ag = Float64[]
     for sp in field
        for leaf in apply(sp,Query(Main.Plant.Leaf))
            leaf.fabs=(leaf.PARperday)/ref
            data(sp).Ag += leaf.Ag*leaf.area
         println("Amax $(leaf.Amax) LNarea $(leaf.LNarea) area $(leaf.area) fabs $(leaf.fabs) PAR $(leaf.PARperday)")
         end
        for pet in apply(sp,Query(Main.Plant.Petiole))
            pet.fabs=(pet.PARperday)/ref
         data(sp).Ag += pet.Ag*pet.area
         end    
        for int in apply(sp,Query(Main.Plant.Internode))
            int.fabs=(int.PARperday)/ref
         data(sp).Ag += int.Ag*int.area
         end
       # push!(Ag, data(sp).Ag)
     end
     # mol/plant
 end



# function reset_PAR!(all_leaves,all_petiole,all_internodes,vars)
#      vars.PAR = 0.0
#        for l in all_leaves
#            l.PAR=0.0
      
#        end
#          for pet in all_petiole
#               pet.PAR=0.0
#          end
#             for int in all_internodes
#                 int.PAR=0.0
#             end

#    return nothing
# end
end