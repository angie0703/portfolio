#=
Version: 1.0.0
Creator: Angela Losu
Creation Date: 2025-12-01
Code description:
This module defines the data structures and parameters specific to the potato species within the Virtual Plant Lab framework.

Credit to Jie Lu for the original model code.
=#

module Plant
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

include("Params.jl")
import .speciestype
include("Growth.jl")
using .Growth
include("Nuptake.jl")
using .Nuptake
include("Nallocation.jl")
using .Nallocation
 
potato_param = speciestype.load_config("C:/Users/angel021/VSCODE/potato/potatoparam.json")

# const GLOBAL_CONFIG_spparams = Ref{speciestype.spparams}()
GLOBAL_CONFIG_spparams = Ref{speciestype.spparams}()
function sp(param)
rootbud_param, rootNode_param, emgrootbud_param,lrootNode_param,shootrootNode_param, BudNode_param, 
rootBudNode_param, meristem_param, rootmeristem_param, lrootmeristem_param, 
rootseg_param, Internode_param, Leaf_param, Petiole_param, tuber_param, Root_param,
spparams_param=param
return rootbud_param, rootNode_param, emgrootbud_param, lrootNode_param, shootrootNode_param, BudNode_param, 
rootBudNode_param, meristem_param, rootmeristem_param, lrootmeristem_param, 
rootseg_param, Internode_param, Leaf_param, Petiole_param, tuber_param, Root_param,
spparams_param
end

  rootbud=typeof(speciestype.rootbud())
    rootNode=typeof(speciestype.rootNode())
   emgrootbud=typeof(speciestype.emgrootbud())
    lrootNode=typeof(speciestype.lrootNode())
    shootrootNode=typeof(speciestype.shootrootNode())
    BudNode=typeof(speciestype.BudNode())
    rootBudNode=typeof(speciestype.rootBudNode())
    meristem=typeof(speciestype.meristem())
    rootmeristem=typeof(speciestype.rootmeristem())
    lrootmeristem=typeof(speciestype.lrootmeristem())
    
    Rootseg=typeof(speciestype.Rootseg())
   Internode=typeof(speciestype.Internode())
    Leaf=typeof(speciestype.Leaf())
    Petiole=typeof(speciestype.Petiole())
    tuber=typeof(speciestype.tuber())
    Root=typeof(speciestype.Root())   
  
 function VirtualPlantLab.feed!(turtle::Turtle, i::Internode, vars) 

    if i.rank==0
         i.branch_angle=0
       else
         i.branch_angle=vars.branch_angle
        end
         #rv!(turtle,i.length*80.236*0.05*i.width*1000)
         ra!(turtle, i.branch_angle)
     if i.order>vars.max_shortint
         rh!(turtle, vars.phyllotaxis)
     else
         rh!(turtle, vars.lowerphyllotaxis)
     end
 
     HollowCylinder!(turtle,move=true,length=i.length, height=i.width, width=i.width,colors=RGB(0.5,0.4,0.0),materials=i.material)
     ra!(turtle, -i.branch_angle)
     return nothing
     end

  
    
     function VirtualPlantLab.feed!(turtle::Turtle, g::tuber, vars)
         HollowCylinder!(turtle, move=true, length=g.length, height=g.width, width=g.width, colors=RGB(0.5,0.4,0.0),materials=g.material)
         #Mesh!(turtle, hc, colors=copy(g.colors), materials=g.material,move=true)
         return nothing
         end    
    
    function VirtualPlantLab.feed!(turtle::Turtle, l::Leaf, vars)
         # Rotate turtle around the arm for insertion angle
         l.angle=vars.leaf_angle/(1+exp(-0.03*(l.ageD-0.5*0.7*vars.leaf_expansion)))
         ra!(turtle, -l.angle)
         #seg=Int(100)
          frBio=0.01* round(100 * l.biomass / vars.potentialbiom_leaf)
          seg=max(50,Int(round(frBio*100,digits=0)))
         #seg=Int(trunc(l.length/0.001))
         #bending angle
         t_pos=pos(turtle)
         t_head=head(turtle)
         center=t_pos.+0.5*l.length*t_head
         l.height=center[3]
         maxS=vars.leaf_Curve/seg
        bendingangleSegment=maxS/(1+exp(-0.03*(l.ageD-0.5*0.7*vars.leaf_expansion)))
       
         # Generate the leaf
         for i in 1:seg
        l.leafsegWidth=0.5*(l.width)*(((1-(1-i/seg))/(1-vars.max_width))*((1-i/seg)/vars.max_width)^(vars.max_width/(1-vars.max_width)))^vars.shapeCoeff
         #l.normarea +=(((1-(1-i/seg))/(1-vars.max_width))*((1-i/seg)/vars.max_width)^(vars.max_width/(1-vars.max_width)))^vars.shapeCoeff*0.01
         #l.area+=l.leafsegWidth*l.length/seg
         Rectangle!(turtle, move=true, length = l.length/seg, width =2*l.leafsegWidth,colors=RGB(0.1,0.5,0.2),materials=l.material)
                  if i==seg/2
        ra!(turtle, -20*bendingangleSegment) 
         end
                  end        
         # Rotate turtle back to original direction
  
         ra!(turtle, l.angle)
        return nothing

     end

     function VirtualPlantLab.feed!(turtle::Turtle, p::Petiole, vars)
         # Rotate turtle around the arm for insertion angle
        p.angle=vars.leaf_angle/(1+exp(-0.03*(p.ageD-0.5*0.7*vars.leaf_expansion)))
         ra!(turtle, -p.angle+12.5)
         # Generate the leaf
         Rectangle!(turtle,move=true,length = 0.1, width = 0.0025,colors=RGB(0.1,0.5,0.2),materials=p.material )
        
         ra!(turtle, p.angle-12.5)
         return nothing
    
       
     end
    
     function VirtualPlantLab.feed!(turtle::Turtle, rs::Rootseg, vars)
        if rs.order==0 
         #insertion angle for axial root
        if rs.rank != 0
             rs.axialroot_angle = vars.axialroot_angle + rand(-10.0:10.0)
         else
            rs.axialroot_angle = 180.0 + rand(-10.0:10.0)
         end
         rs.IBD = vars.IBD
        ra!(turtle, rs.axialroot_angle)
                  if vars.species == "potato" && rs.rank != 0
             rv!(turtle, -rs.IBD * 80.236 * 0.05 * rs.width * 1e3)
         end   
         # generate a lateral root
        elseif rs.order==1
                     rh!(turtle,rs.topangle)
         rs.lateral_angle=vars.lateral_angle+rand(-40.0:0)
         ra!(turtle,-rs.lateral_angle)
        end


        t_pos = pos(turtle)
         t_head = head(turtle)
        center = t_pos .+ 0.5 * vars.IBD * t_head
        rs.x = center[1]
        rs.y = center[2]
         rs.z = center[3]

         # Periodic boundary conditions for x and y
           if Main.infinite==true
         apply_periodic_boundary!(turtle,rs, Main.field_length, Main.field_width) 
    end
         
   
         HollowCylinder!(turtle, move=true, length=rs.IBD, height=rs.width, width=rs.width, colors=rs.colors, materials=rs.material)
        if rs.order==0
         ra!(turtle, -rs.axialroot_angle)
        
        elseif rs.order==1
          ra!(turtle,rs.lateral_angle)
          rh!(turtle, -rs.topangle)
        end
         return nothing
     end
        
         function VirtualPlantLab.feed!(turtle::Turtle, rb::rootbud, vars)
             t_pos=pos(turtle)
             t_head=head(turtle)
             center=t_pos.+0.5*0.000001*t_head
             rb.x=center[1]
             rb.y=center[2]
             rb.z=center[3]
             
            HollowCylinder!(turtle,length=0.000001, height=0.000001, width=0.000001,colors=RGB(0.5,0.4,0.2),materials=rb.material)
         end

         function VirtualPlantLab.feed!(turtle::Turtle, ebb::emgrootbud, vars)
       
            t_pos=pos(turtle)
             t_head=head(turtle)
             center=t_pos.+0.5*0.000001*t_head
             ebb.x=center[1]
             ebb.y=center[2]
             ebb.z=center[3]
            HollowCylinder!(turtle,move=true,length=0.000001, height=0.000001, width=0.000001,colors=RGB(0.5,0.4,0),materials=ebb.material)
            return nothing
         end


     function VirtualPlantLab.feed!(turtle::Turtle, rb::rootBudNode, vars)
          #turtle.message
         # Rotate turtle around the arm for insertion angle
          if rb.val!=0
          rh!(turtle,60.0*rb.val+rand(-30:30.0))
          else
          rh!(turtle,180.0)
          end

      
         HollowCylinder!(turtle,length=0.0, height=0.0, width=0.0,colors=RGB(0.5,0.4,0),materials=rb.material)
         return nothing
     end
    
     # Insertion angle for the bud nodes
    function VirtualPlantLab.feed!(turtle::Turtle, b::BudNode, vars)
         rh!(turtle,(60.0)*b.val+rand(-20.0:20.0))
         return nothing
     end

     function VirtualPlantLab.feed!(turtle::Turtle, rm::rootmeristem,vars)
         # Rotate turtle around the arm for insertion angle
         t_pos=pos(turtle)
        t_head=head(turtle)

        center=t_pos.+0.5*0.00001*t_head
         rm.x=center[1]
         rm.y=center[2]
         rm.z=center[3]
       
         SolidCylinder!(turtle,length=0.00001, height=0.00001, width=0.00001,move=true, colors=RGB(0.5,0.4,0), materials = rm.material)
         return nothing
        
     end

     function VirtualPlantLab.feed!(turtle::Turtle, rn::rootNode,vars)
         t_pos=pos(turtle)
         t_head=head(turtle)
         center=t_pos.+0.5*0.0000001*t_head
        rn.x=center[1]
         rn.y=center[2]
        rn.z=center[3]
        SolidCylinder!(turtle,length=0.0000001, height=0.0000001, width=0.0000001, move=true, colors=RGB(0.5,0.4,0), materials = rn.material)
       
     return nothing
     end
    
     function VirtualPlantLab.feed!(turtle::Turtle, lrm::lrootmeristem, vars)
         # Rotate turtle around the arm for insertion angle
        
          t_pos=pos(turtle)
         t_head=head(turtle)
         center=t_pos.+0.5*0.00001*t_head
         lrm.x=center[1]
         lrm.y=center[2]
         lrm.z=center[3]
         SolidCylinder!(turtle,length=0.00001, height=0.00001, width=0.00001, move=true, colors=lrm.colors, materials = lrm.material)
          ra!(turtle, -vars.lateral_angle)
         nothing
     end

      function VirtualPlantLab.feed!(turtle::Turtle, lrn::lrootNode,vars)

          t_pos=pos(turtle)
          t_head=head(turtle)
          center=t_pos.+0.5*0.0000001*t_head
          lrn.x=center[1]
     lrn.y=center[2]
          lrn.z=center[3]
          SolidCylinder!(turtle,length=0.0000001, height=0.0000001, width=0.0000001, move=true, colors=RGB(0.5,0.4,0), materials = lrn.material)
      return nothing
     end

 

 #construct leaflet (leaf +petiole)   
function leaflet(mer,vpet,vleaf)
    leaflet= Petiole(biomass = vpet.biomass,
                                               length  = vpet.length,
                                               width   = vpet.width,order=data(mer).cum_phytomer+1, rank=data(mer).rank)+Leaf(biomass = vleaf.biomass,
                                               length  = vleaf.length,width = vleaf.width,ageD=vleaf.ageD,order=data(mer).cum_phytomer+1,rank=data(mer).rank)
    
return leaflet                                           
end


function create_meristem_rule(vleaf,vpet, vint,vtuber,varoot)
   #root enlongation
    function rootenlongate(node)
        val=graph_data(node).val_root
        out =Rootseg(width=varoot.width,val=val+1,rank=data(node).rank,order=0)
        # function to determine if a new lateral root initiation or not based on the auxin
        if graph_data(node).pinit==true
        initprob=0.161*data(node).totalauxin*exp(-0.128*data(node).totalauxin)+0.491
        else
            initprob=1
        end
        data(node).initprob=initprob
        #generate axial root elongation
        for i in 1:data(node).rootseg-1
            data(node).r=rand()
            if data(node).r<initprob
            out = out + (rootbud(rank=data(node).rank,fracshootaux=data(node).fracshootaux),) + Rootseg(val=i+val+1,rank=data(node).rank,biomass=varoot.biomass,width=varoot.width,order=data(node).order)
            else
            out =out+Rootseg(val=i+val+1,rank=data(node).rank,biomass=varoot.biomass,width=varoot.width,order=data(node).order)
        end
end
    out = out + rootNode(rank=data(node).rank,order=data(node).order)
    return out
end

#lateral root enlongation
    function lrootenlongate(mer)
        out = speciestype.Node()
        for i in 1:data(mer).rootseg
            out = out + Rootseg(rank=data(mer).rank,order=data(mer).order)
        end
        if data(mer).z < 0.0001
            out = out + lrootNode(rank=data(mer).rank,order=1)
        end
        return out
    end
     
   #shootenlongation criteria
    function shootenlongation(mer)
        return  graph_data(mer).age>graph_data(mer).growth_start&&data(mer).ageD>graph_data(mer).plastochron
    end
#shoot enlogantion                                                                                                   
    function consphytomer(mer) 
        if graph_data(mer).species=="potato"
           if data(mer).cum_phytomer<graph_data(mer).max_leaf+1
                      out = speciestype.Node()+(Internode(biomass = vint.biomass,
                            length  = vint.length, width = vint.width,order=data(mer).order+1, rank=data(mer).rank)+
                            meristem(order=data(mer).order+1,rank=data(mer).rank,cum_phytomer=data(mer).cum_phytomer+1),
                            leaflet(mer,vpet,vleaf))
         
            else
             out=tuber(biomass=vtuber.biomass,length=vtuber.length,width=vtuber.width)
            end
        else
          
               if data(mer).cum_phytomer>graph_data(mer).max_leaf+1
                out= speciestype.Node()+ (Internode(biomass = vint.biomass,
                length = vint.length, width = vint.width,order=data(mer).cum_phytomer+1,rank=data(mer).rank)+
                 meristem(order=data(mer).order+1,rank=data(mer).rank,cum_phytomer=data(mer).cum_phytomer+1), 
                 (tuber(biomass=vtuber.biomass,length=vtuber.length,width=vtuber.width),
                 leaflet(mer,vpet,vleaf)))
                else
                        out = speciestype.Node()+ (Internode(biomass = vint.biomass,
                               length = vint.length, width = vint.width,order=data(mer).cum_phytomer+1,rank=data(mer).rank)+
                               meristem(order=data(mer).order+1,rank=data(mer).rank,cum_phytomer=data(mer).cum_phytomer+1),
                               leaflet(mer,vpet,vleaf))
                end
        end
            return out
           
         end
                           
         phytomer(mer) = consphytomer(mer)
#meristem rules
#shoot enlongation         
         meristem_rule=Rule(meristem, lhs =shootenlongation,rhs =mer->phytomer(mer))
#remove leaf and petiole after achieve their life span
       
        remove_leaf=Rule(Leaf,lhs=leaf->data(leaf).ageD>graph_data(leaf).leaf_expansion*4||(data(leaf).LNarea<0.3&&data(leaf).ageD>0.5*graph_data(leaf).leaf_expansion),
                              rhs=leaf->nothing)

        remove_petiole=Rule(Petiole,lhs=petiole->data(petiole).ageD>graph_data(petiole).leaf_expansion*4,
                              rhs=petiole->nothing)                    
#root enlongation
        rootmeristem_rule=Rule(rootNode,lhs=rootnode->data(rootnode).rootseg>0,rhs= node -> rootenlongate(node))
#lateral root enlongation
         lateralrootmeristem_rule=Rule(lrootNode, rhs=mer->lrootenlongate(mer) )
         
         return meristem_rule, rootmeristem_rule, lateralrootmeristem_rule, remove_leaf,remove_petiole
                              end

    function shootrootbranching(rootnode)
                                        newroot::Bool=false
        if graph_data(rootnode).rank_root< graph_data(rootnode).max_root&&graph_data(rootnode).species=="potato"
                    if data(rootnode).er > 1
                    while data(rootnode).er > 1   
                    data(rootnode).er -= 1
                    data(rootnode).n += 1
                    end
                    newroot=true
                    else
                        newroot=false
                    end
            
            else 
                newroot= false
            end
            
            return newroot
    end


function create_branch_rule(vrb,vrootmer,mer,vars,vmer, vbn, verb)
   
   #shoot branching  
    function createbranch(node)
        new_BudNode=BudNode(rank=data(node).rank+1,val=data(node).val+1,order=vbn.order)
         out=(new_BudNode+
         meristem(rank=data(node).rank+1, order=vmer.order))+BudNode(rank=data(node).rank,val=data(node).val,order=vbn.order)
        return out
      end  
    #tiller initiation
    function dormaince(node)
        if graph_data(node).species=="potato"
           if  has_descendant(node, condition= n-> data(n) isa meristem)[1]
             m=get_descendant(node, condition= n->data(n) isa meristem && !(children(n) isa BudNode))
               if ismissing(m)
                 return false  
              else
                  if data(m).cum_phytomer-data(node).rank>data(node).dom&&rand()>0.3&&graph_data(node).tiller<graph_data(node).maxtiller&&data(m).ageD>graph_data(node).plastochron
                     return true
                       else
                      return false
                   end
              end
          else
             return false
          end
        else 
            return false
        end
    end
    
    function transfer(emgrootbud)
        if has_ancestor(emgrootbud, condition=n->data(n) isa Rootseg)[1]
           return (true, (ancestor(emgrootbud),))
        else
           return (false, ())
        end
       end
       
 #axial root branching criteria
    #lateral root branching
    function lateral_branch(rootbud)
        if shootauxintrans(rootbud)
            return emgrootbud(rank=data(rootbud).rank)+RH(rand(0:360.0))+lrootNode(rank=data(rootbud).rank,order=1)+lrootmeristem(rank=data(rootbud).rank,order=1)
        else
            return speciestype.Node()
        end
    end

   #lateral root branching emergence criteria (shoot auxin transportation)
    function shootauxintrans(rootbud)
        p1=parent(rootbud)
        check = has_descendant(p1, condition = n -> data(n) isa rootmeristem)[1]
        if check   
                fracshootaux = data(rootbud).fracshootaux
                if graph_data(rootbud).pemerge==true
                prob = max(0.0286, (1 - 1.033 * exp(-0.212 * fracshootaux*10)))
                else
                    prob=1
                end
                #println("$(data(rootbud).rank)aux$(fracshootaux)prob$(prob)")
                return rand() < prob
        else
        return false
        end
    end
 
       
   #shoot branching rule 
    branch_rule=Rule(BudNode,lhs=dormaince,rhs = node->createbranch(node))
   #root branching rule
    shootrootbranch_rule=Rule(shootrootNode,lhs=rootnode->shootrootbranching(rootnode)[1], rhs= rootnode -> shootrootNode(val=data(rootnode).val+1,rank=graph_data(rootnode).rank_root)+
                                                        Tuple(rootBudNode(ageD=vrb.ageD,rank=data(rootnode).rank+1,val=data(rootnode).val+1)+rootNode(rank=data(rootnode).rank+1,val=data(rootnode).val+1,order=0)+
                                                        rootmeristem(val=vrootmer.val+1,rank=data(rootnode).rank+1,order=0) for i in 1:data(rootnode).n ))
   #lateral root branching rule 
   rootbranch_rule=Rule(rootbud,rhs= rootbud ->lateral_branch(rootbud) )  
   transfer_rule=Rule(emgrootbud,lhs=transfer,rhs=(erb,aroot)->emgrootbud(val=data(aroot).val,rank=data(aroot).rank),captures=true)

   return rootbranch_rule, shootrootbranch_rule,branch_rule,transfer_rule
end


 function apply_periodic_boundary!(turtle,rootseg, fieldlength, fieldwidth)
     wrapped = false
     # Wrap x coordinate
     if rootseg.x < 0
         rootseg.x += fieldlength
         wrapped = true
     elseif rootseg.x >= fieldlength
         rootseg.x -= fieldlength
         wrapped = true
     end
     # Wrap y coordinate
     if rootseg.y < 0
         rootseg.y += fieldwidth
         wrapped = true
     elseif rootseg.y >= fieldwidth
         rootseg.y -= fieldwidth
         wrapped = true
     end
         # If it's an arootseg and wrapped, reposition the rootseg in the graph by updating its coordinates
         if wrapped==true
             # Find the node in the graph and update its coordinates
             if !ismissing(rootseg)
                 if hasproperty(rootseg, :x)
                     rootseg.x = rootseg.x
                 end
                 if hasproperty(rootseg, :y)
                     rootseg.y = rootseg.y
                 end
                
             end
             t!(turtle,to=Vec(rootseg.x, rootseg.y, rootseg.z)) # Update the turtle's position
         end
     return nothing
 end


function daily_step!(field,soil,DOY,DL,i)
    #Main.create_scene(field)
    @threads for species in field
        vars =  data(species)
        all_soils=Main.get_soils(soil)
        # Retrieve all the relevant organs
        all_leaves =apply(species, Query(Leaf))
        all_internodes = apply(species, Query(Internode))
        # all_tubers=apply(species,Query(tuber))
        all_petiole=apply(species, Query(Petiole))
        
        all_meristems =apply(species, Query(meristem))
        all_rootmeristem =apply(species, Query(rootmeristem))
        all_rootNodes=apply(species, Query(rootNode, condition= n->data(n).order==0))
        all_lrootNodes=apply(species, Query(lrootNode,condition= n->data(n).order==1))
        all_lrootmeristem=apply(species, Query(lrootmeristem))
        all_rootBudNode=apply(species, Query(rootBudNode))
        all_shootrootNode=apply(species, Query(shootrootNode))
        all_rootbud =apply(species,Query(rootbud))
        all_roots=apply(species,Query(Rootseg))
        all_budnodes=apply(species,Query(BudNode))
        all_emgrootbud=apply(species,Query(emgrootbud))
        all_aroots=apply(species,Query(Rootseg, condition= n->data(n).order==0))
        vars.leafnumber=length(all_leaves)
        vars.rootnumber=length(all_emgrootbud) #total number of lateral roots
        vars.tiller=length(all_budnodes) #total number of tillers
        println("tiller$(vars.tiller)")
        vars.rank_root=length(all_rootmeristem) #total number of axial root
        
        vars.lateralrootnumber=length(all_emgrootbud)
        println("$(vars.species)_plant#$(vars.plant_number) $(vars.species)_row#$(vars.row)")
         if i> vars.sowing_delay 
        # Update the age of the organs
        Growth.age!(all_leaves, all_internodes, all_tubers,all_meristems, all_petiole,all_rootmeristem, all_lrootmeristem,all_rootBudNode,all_shootrootNode,vars,DOY)
        # Grow the plant
        
          
        Growth.grow!(DL,species,all_leaves, all_internodes, all_tubers ,all_petiole,all_rootmeristem,all_lrootmeristem,all_roots,all_aroots,all_rootbud,all_meristems,all_budnodes,all_emgrootbud)
        Nuptake.Nfix!(vars,all_roots)
        Nuptake.exploresoil!(all_roots,all_rootmeristem,all_lrootmeristem,all_soils,vars)
        Growth.SAScoefficient!(vars)
        Growth.size_leaves!(all_leaves, vars)
        Growth.size_internodes!(all_internodes, vars)
        Growth.size_tubers!(all_tubers, vars)
        Growth.size_petiole!(all_petiole,vars)
        Growth.size_rootmer!(all_rootmeristem,all_rootNodes,vars)
        Growth.size_lrootmer!(all_lrootmeristem,all_lrootNodes,vars)
        Growth.size_root!(all_roots,vars)
        
         if Main.raytrace==true
         Nallocation.N_sinkstrength!(species,all_leaves,all_internodes,all_petiole,all_tubers)
        Nallocation.N_allocation!(all_leaves,all_petiole,all_internodes,all_tubers,vars)
        end
       Growth.reset_sink!(all_leaves,all_internodes,all_tubers,all_petiole,all_rootmeristem,all_lrootmeristem,vars)
        #reset_auxp!(field)
        Nallocation.reset_Nsink!(vars,all_leaves,all_petiole,all_internodes,all_tubers)
        Nallocation.reset_N!(vars,all_leaves,all_petiole,all_internodes)
        #Growth.reset_PAR!(all_leaves,all_petiole,all_internodes,vars)
      
    end
        #create a scene
        # Developmental rules
        rewrite!(species)
    end
    
end

function create_plant(origin,plant_number::Int64,row::Int64,orientation,param)
    species=sp(param)
    GLOBAL_CONFIG_spparams[]=species[end]
  function spparams(; kwargs...)
    return GLOBAL_CONFIG_spparams[]
  end
    # Initial state and parameters the plant
     vars = spparams()
     vars.plant_number=plant_number
     vars.row=row 
     vars.totalN=vars.N0   
    leaf=Leaf()
    int=Internode()
    #tuber=typeof(tuber())
    # Initial states of the leaves
    leaf_length, leaf_width, leaf_area = Growth.leaf_dims(vars.LB0*0.95, vars, leaf)
    vleaf = (biomass = vars.LB0*0.95, length = leaf_length, width = leaf_width,area=leaf_area,ageD=leaf.ageD, rank=0, order=0)
    # Initial states of the petiole
    petiole_length, petiole_width, petiole_area = Growth.pet_dims(vars.LB0*0.05, vars)
    vpet = (biomass = vars.LB0*0.05, length = petiole_length, width = petiole_width,area=petiole_area,order=0,rank=0)

    # Initial states of the internodes
    int_length, int_width,int_area = Growth.int_dims(vars.IB0, vars,int)
    vint = (biomass = vars.IB0, length = int_length, width = int_width,area=int_area,rank=0,order=vars.max_shortint+1)
    #Initial states of the tuber
    g_length, g_width = Growth.tuber_dims(vars.IB0, vars,tuber)
    vtuber = (biomass = vars.IB0, length = g_length, width = g_width)
    #Initial state of root
    rootmer=speciestype.rootmeristem()
    rootmer_length, rootmer_width,rootmer_rootseg=Growth.rootmer_dims(rootmer.Δarootbiom, vars, rootmer)
    vrootmer=(biomass=vars.IB0,length=rootmer_length,width=rootmer_width,rootseg=rootmer_rootseg,val=rootmer.val_root,colors=rootmer.colors,rank=rootmer.rank)
     
    bn=BudNode()
    erb=emgrootbud()
    verb=()
    vbn=(rank=0,val=0,order=0)
  
    lrootmer=lrootmeristem()
    lrootmer_length,lrootmer_width, lrootmer_rootseg=Growth.lrootmer_dims(lrootmer.Δlrootbiom, vars, lrootmer)
    vlrootmer=(biomass=lrootmer.Δlrootbiom,length=lrootmer_length,width=lrootmer_width,rootseg=lrootmer_rootseg,colors=lrootmer.colors,rank=erb.rank)
    rb=rootBudNode()
    vrb=(ageD=rb.ageD,val=rb.val,rank=vars.rank_root)
    root=Rootseg()
   #
    mer=meristem()
    vmer=(rank=0,phytomer_distance=mer.phytomer_distance,cum_phytomer=0,order=0)
    
   
    root_length,root_width=Growth.root_dims(vars.RTD*pi*(vars.width/2)^2*vars.IBD*1e3,vars,root)
    varoot=(biomass=root.biomass,length=root_length, width=root_width, val=rootmer.val_root,order=0)
    
    # Growth rules
    meristem_rule = create_meristem_rule(vleaf,vpet,vint,vtuber,varoot)
    rootbranch_rule   = create_branch_rule(vrb,vrootmer,mer,vars,vmer,vbn,verb)
   
             axiom =T(origin)+RH(orientation)+(RH(90.0)+BudNode(rank=vbn.rank,val=vbn.val,order=vbn.order)+
             meristem(rank=vmer.rank,phytomer_distance=vmer.phytomer_distance,cum_phytomer=0,order=vmer.order),
             shootrootNode()+rootBudNode(ageD=vrb.ageD,val=vrb.val)+
             rootNode(order=0)+rootmeristem(biomass=vrootmer.biomass,length=vrootmer.length,width=vrootmer.width,val=vrootmer.val,colors=vrootmer.colors))
    if Main.leafremoval==true
    species = Graph(axiom = axiom, rules = (meristem_rule[1],meristem_rule[2],meristem_rule[3],meristem_rule[4],meristem_rule[5],
    rootbranch_rule[1],rootbranch_rule[2],rootbranch_rule[3],rootbranch_rule[4]),data=vars)
    else
    species = Graph(axiom = axiom, rules = (meristem_rule[1],meristem_rule[2],meristem_rule[3],
    rootbranch_rule[1],rootbranch_rule[2],rootbranch_rule[3],rootbranch_rule[4]),data=vars)
    end
    return species
end


end