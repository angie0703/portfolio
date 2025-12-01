#=
Version: 1.0.0
Creator: Angela Losu
Creation Date: 2025-12-01
Code description:
This module defines the data structures and parameters specific to the potato species within the Virtual Plant Lab framework.

Credit to Jie Lu for the original wheat model code.
=#

# using JSON

module speciestype
import VirtualPlantLab as VPL
import ColorTypes:RGB
import Ecophys
using Distributions
using NamedArrays
import Parameters:@with_kw
using JSON
struct bud<: VPL.Node end
@with_kw mutable struct rootbud<: VPL.Node
x::Float64=0.0
y::Float64=0.0
z::Float64=0.0
fracshootaux::Float64 =0.0
rank::Int64=0
order::Int64=0
colors::RGB=RGB(0.5,0.4,0)
material::VPL.Lambertian{1} = VPL.Lambertian(τ = 0.1, ρ = 0.05)
end 

struct Node<: VPL.Node end
@with_kw mutable struct rootNode<:VPL.Node 
   
rootseg::Int64 = 0
x::Float64=0.0
y::Float64=0.0
z::Float64=0.0
val::Int64=0
order::Int64=0
rank::Int64=0
totalauxin::Float64=0.0
fracshootaux::Float64=0.0
initprob::Float64=0.0
r::Float64=0.0
material::VPL.Lambertian{1} = VPL.Lambertian(τ = 0.1, ρ = 0.05)
colors::RGB=RGB(0.5,0.4,0)
end


# Emergence of lateral root bud
@with_kw mutable struct emgrootbud<: VPL.Node 
    x::Float64=0.0
    y::Float64=0.0
    z::Float64=0.0
    fracshootaux::Float64 =0.0
    rootseg::Int64 = 0
    rank::Int64=0
    val::Int64=0
    material::VPL.Lambertian{1} = VPL.Lambertian(τ = 0.1, ρ = 0.05)
    end



 @with_kw mutable struct lrootNode<: VPL.Node
      rootseg::Int64 = 0
      x::Float64=0.0
      y::Float64=0.0
      z::Float64=0.0
      val::Int64=0
      rank::Int64=0
      order::Int64=0
      material::VPL.Lambertian{1} = VPL.Lambertian(τ = 0.1, ρ = 0.05)
      colors::RGB=RGB(0.5,0.4,0)
     end

   

    @with_kw mutable struct shootrootNode<:VPL.Node 
    age::Int64= 0
    ageD::Float64= 0
    val::Int64=0
    rank::Int64=0   #split root: 1, other normal scenario:0
    order::Int64=0
    er::Float64=0
    n::Int64=0
    colors::RGB=RGB(0.5,0.4,0)
    end
   
@with_kw mutable struct BudNode<: VPL.Node 
    number::Int64=0
    age::Int64= 0
    ageD::Float64= 0
    ageDprevious::Float64=0.0   
    dom::Float64=0.0
    sr::Float64=0.0
    tiller::Float64=0.0
    phytomer_distance::Float64 = 0.0
    rank::Int64=0
    order::Int64=0
    cum_phytomer::Int64 = 0
    val::Int64=0
    branch_angle::Float64=0.0
     material::VPL.Lambertian{1} = VPL.Lambertian(τ = 0.1, ρ = 0.05)
end


@with_kw mutable struct rootBudNode<: VPL.Node
    age::Int64= 0
    ageD::Float64= 0
    ageDprevious::Float64=0.0   
    val::Int64=0
    rank::Int64=0
    order::Int64=0
     material::VPL.Lambertian{1} = VPL.Lambertian(τ = 0.1, ρ = 0.05)
    end

    

    @with_kw mutable struct rootmeristem<: VPL.Node 
        age::Int64= 0
        ageD::Float64= 0
        ageDprevious::Float64=0.0
        rootseg::Int64=0
        val:: Int64=0
        val_root::Int64=0
        order::Int64=0
        rank::Int64=0
        x::Float64=0.0
        y::Float64=0.0
        z::Float64=0.0
        length::Float64=0.000
        width::Float64=0.0
        totallength::Float64=0.0
        difflength::Float64=0.00
        biomass::Float64 =0
        biomassprevious::Float64 =0
        total_sink::Float64 =0
        aroot_sinkperday::Float64 = 0.0
        shootauxin::Float64 =0
        
        merauxinN::Float64 =0
        totalmerauxin::Float64=0
        totalauxin::Float64=0
        totalauxinperday::Float64=0
        ELN::Float64=0
        Δshootauxin::Float64 =0
        fracshootaux::Float64=0
        decay::Float64 =0
        merauxcoef::Float64=0
        coef::Float64=1
     Δarootbiom::Float64= 0    
     colors::RGB=RGB(0,0,1)

     material::VPL.Lambertian{1} = VPL.Lambertian(τ = 0.1, ρ = 0.05) 
end



@with_kw mutable struct lrootmeristem<: VPL.Node 
            age::Int64= 0
            ageD::Float64= 0
            ageDprevious::Float64=0.0
            rootseg::Int64=0
            por::Int64=0
            val:: Int64=0
            rank::Int64 =0
            order::Int64 =0
            x::Float64=0.0
            y::Float64=0.0
            z::Float64=0.0
        length::Float64=0.000
        width::Float64=0.0
        biomass::Float64 =0.00
        biomassprevious::Float64 =0.00
        total_sink::Float64 =0
        lroot_sinkperday::Float64 = 0.0
        finerootsink::Float64 =0.0
        fineroot_sinkperday::Float64 = 0.0
        finerootbiomass::Float64 =0.0
        totallength::Float64=0.0
        difflength::Float64=0.00
        shootauxin::Float64 =0
        merauxinN::Float64 =0
        totalmerauxin::Float64=0
        totalauxin::Float64 = 0.0
        totalauxinperday::Float64=0
        coef::Float64=100  #auxin sensitivity coefficient on root elongation between primary root and lateral root 
        decay::Float64 =0
        merauxcoef::Float64=0
        NRTcoef::Float64=0
        ELN::Float64=0
     Δlrootbiom::Float64= 0
     Δshootauxin::Float64 =0
     colors::RGB=RGB(totalauxin/10,0,1-totalauxin/10)
     material::VPL.Lambertian{1} = VPL.Lambertian(τ = 0.1, ρ = 0.05)
            end



@with_kw mutable struct meristem<: VPL.Node 
    age::Int64= 0
ageD::Float64= 0
ageDprevious::Float64=0.0
agewait::Float64 =0.0
leafnumber::Float64 = 0.0
cum_phytomer::Int64=0
rank::Int64 =0
order::Int64 =0
dom::Float64=0.0
parent_rank::Int64=0
phytomer_distance::Float64=0.0
x::Float64=0.0
y::Float64=0.0
z::Float64=0.0
     material::VPL.Lambertian{1} = VPL.Lambertian(τ = 0.1, ρ = 0.05)
end



@with_kw mutable struct Internode<: VPL.Node
  
     age::Int64 = 0
     ageD::Float64 = 0.0
     ageDprevious::Float64=0.0
     agewait::Float64 =0.0
     rank::Int64=0
     order::Int64=0
     biomass::Float64 = 0.00
     Δbiomass::Float64=0
     biomassprevious::Float64=0.00
     potentialbiom::Float64=0
     length::Float64 = 0.0
     width::Float64 = 0.0
     area::Float64=1e-6
     absm::Float64=0 #absorbed radiation per unit area
     fabs::Float64 = 0 #fraction of globRad absorbed
     PAR::Float64=0.0
     PARdif::Float64 = 0.0
    PARdir::Float64 = 0.0
    PARperday::Float64=0.0
     DLNarea::Float64 = 0.0 #nitrogen demand per specific leaf area
     DLN::Float64=0.0 #nitrogen demand
     Navaliable::Float64=0.0
     LNarea::Float64=0.0
     Amax::Float64 = 0
     material::VPL.Lambertian{1}=VPL.Lambertian(τ = 0.1, ρ = 0.05)
     branch_angle::Float64=0.0
     #sink :: Exponential{Float64} = Exponential(1)
     Ag::Float64 = 0.0
     total_sink::Float64 =0
     Nsink::Float64=0
     end

        @with_kw mutable struct Leaf<:VPL.Node
     age::Int64 = 0
     ageD::Float64 = 0.0
     ageDprevious::Float64=0.0
     agewait::Float64 =0.0
     rank::Int64=0
     order::Int64=0
     biomass::Float64 = 0.0
        Δbiomass::Float64=0
        biomassprevious::Float64=0.0
     length::Float64 = 0.00
     width::Float64 = 0.0
     height::Float64=0.0
     total_sink::Float64 =0
     Nsink::Float64=0
     absm::Float64=0 #absorbed radiation per unit area
     fabs::Float64 = 0 #fraction of globRad absorbed
     #fracblade::Float64 =0.95
     #sink::Beta{Float64}=Beta(2,5)
     PARdif::Float64 = 0.0
    PARdir::Float64 = 0.0
    PARperday::Float64=0.0 
    normarea::Float64=1e-6
     leafsegWidth::Float64 = 0.0
     leafsegLength::Float64=0.0
     potentialbiom::Float64=0
     angle::Float64 =30 
     material::VPL.Lambertian{1}= VPL.Lambertian(τ = 0.1, ρ = 0.05)
     Idir_PAR::Float64 = 0.0
     Idif_PAR::Float64 = 0.0
     Ag::Float64 = 0.0
     auxp::Float64=1
     PAR::Float64=0.0
     DLNarea::Float64 = 0.0 #nitrogen demand per specific leaf area
     DLN::Float64=0.0 #nitrogen demand
     Navaliable::Float64=0.0
     LNarea::Float64=0.0
     Amax::Float64 = 0
     area::Float64=1e-6
     end
    
    

     @with_kw mutable struct Petiole<:VPL.Node
     age::Int64 = 0
     ageD::Float64 = 0.0
     ageDprevious::Float64=0.0
     agewait::Float64 =0.0
     difageD::Float64=0.0
     leafnumber::Int64 =0
     biomass::Float64 = 0.00
        Δbiomass::Float64=0
        biomassprevious::Float64=0.00
     length::Float64 = 0.0
     width::Float64 = 0.0
     Nsink::Float64=0
     area::Float64=1e-6
     absm::Float64=0 #absorbed radiation per unit area
     fabs::Float64 = 0 #fraction of globRad absorbed
     PAR::Float64=0.0
     PARperday::Float64=0.0
     PARdif::Float64 = 0.0
    PARdir::Float64 = 0.0
     DLNarea::Float64 = 0.0 #nitrogen demand per specific leaf area
     DLN::Float64=0.0 #nitrogen demand
     Navaliable::Float64=0.0
     LNarea::Float64=0.0
     Amax::Float64 = 0
     angle::Float64=30
     material::VPL.Lambertian{1} = VPL.Lambertian(τ = 0.1, ρ = 0.05)
     Ag::Float64 = 0.0
     total_sink::Float64 =0
     order::Int64=0
     rank::Int64=0
     end

   

     @with_kw mutable struct tuber<:VPL.Node
        age::Int64 = 0
         ageD::Float64 = 0.0
         ageDprevious::Float64=0.0
         biomass::Float64 = 0.00
         biomassprevious::Float64=0.00
         Δbiomass::Float64=0
         potentialbiom::Float64=0
         ΔC::Float64=0
         length::Float64=0
         width::Float64=0.1
         total_sink::Float64 =0
         Nsink::Float64=0
         material::VPL.Lambertian{1} = VPL.Lambertian(τ = 0.1, ρ = 0.05)
         end


         @with_kw mutable struct Root<:VPL.Node
        age::Int64 = 0
         ageD::Float64 = 0.0
         ageDprevious::Float64=0.0
         biomass::Float64 = 0.00
            biomassprevious::Float64=0.00
            Δbiomass::Float64=0
         potentialbiom::Float64=2
         ΔC::Float64=0
         length::Float64 = 0.00
     
     total_sink::Float64 =0
         end

        
         @with_kw mutable struct Rootseg<:VPL.Node
         order::Int64 =0
         rank::Int64 = 0
         width::Float64 = 0.00
         length::Float64 =0.0
         RTD::Float64=60 
         biomass::Float64=0
         biomassradi::Float64=0
         biomassprevious::Float64=0
         val::Int64=1
         val_root::Int64=0
         value::Float64=0
         IBD::Float64 = 0.003
         x::Float64=0.0
         y::Float64=0.0
         z::Float64=0.0
         rootseg::Int64 =0 
         total_sink::Float64 =0.0
         totalarea::Float64=0.0
         auxp::Float64=0.0
         fauxp::Float64=0.0
         potNup::Float64=0.0
        actNup::Float64=0.0
        axialroot_angle::Float64=140
        topangle::Float64 =rand(0.0:10.0)
        finerootbiomass::Float64=0.0
        finerootD::Float64=0.0001
        finerootsink::Float64=0.0
        fineroot_sinkperday::Float64 = 0.0
        lateral_angle::Float64=nor(60,10)
        Nodule::Bool=false
        nodulesink::Float64=0
        nodulebiomass::Float64=0
        colors::RGB=RGB(0.5,0.4,0)
        material::VPL.Lambertian{1} = VPL.Lambertian(τ = 0.1, ρ = 0.05)
         end

      
       
@with_kw mutable struct spparams
    
    plant_number::Int64=0
    strip::Int64=0
    row::Int64=0
    sowing_delay::Float64=0.0
     age::Int64 = 0
    ageD::Float64 = 0.0
    ageDprevious::Float64=0.0
    growth_start::Float64=0 #growth start for shoot( days)
    
    seedbiomass::Float64=0.025
    biomass::Float64 = 0 # Current total biomass (g)
    previousbiomass::Float64= 0
    actbiomass::Float64 = 1e-6
    actbiomass_leaf::Float64 =0
    actbiomass_stem::Float64 =0
    actbiomass_tuber::Float64 =0
    rs::Float64=0
    Cpool::Float64 =0.025
    tb::Float64=0 #base temperature
    rg::Float64=0 #growth respiration in fraction biomass
    total_sink::Float64=0.0
    total_sinkperday::Float64=0.0  #plant carbon sink perday
    total_rootsink::Float64=0.0
    root_sinkperday::Float64=0.0 #root carbon sink perday
    rootNsink::Float64=0.0
    sr::Float64=0
    dom::Float64=0.0
    #photosysthesis related parameter
    Nphoto::Bool=true     #false: disable N related photosynthesis
    totalphotoarea::Float64=0.0 #total photosynthetic area (m2)
    photos::Ecophys.C3{Float64} = Ecophys.C3()
    Ag::Float64=0.0 #daily total photosynthesis (umol)
    
    SLN::Float64=0.0 #average of all specific photosyntic organ nitrogen content(g/m2)
    PAR::Float64=0
    Amax::Float64=0.0 #maximum photosynthesis rate (umol/m2/s)
    alpha::Float64 =0.05   #initial slope of light response curve
    lambda::Float64 =20    # N related Amax parameter
    LN0::Float64 = 1.5     #maximum specific nitrogen (g/m2)
    kNkl::Float64 =0.368 #ratio of nitrogen and light extinction coefficients 
    RUE::Float64 =2.12 #radiation use efficiency (g/MJ) Gou et al.,2017
    totalNsink::Float64=0.0
    totalNsinkstruct::Float64=0.0
    totalNphotosink::Float64=0.0 #total nitrogen sink for photosynthesis
    N0::Float64=seedbiomass*0.03 #initial nitrogen content
    Nc::Float64= 0.0
    allN::Float64=0.0
    fN_internode::Float64=0.01 #fraction of nitrogen in internode (g/g)
    fN_tuber::Float64=0.01 #fraction of nitrogen in tuber (g/g)

    IB0::Float64 = 1e-6  # Initial biomass of an internode (g)
    SIW::Float64 = 0.6   # Specific internode weight (m/g)
    LB0::Float64 = 1e-6  # Initial biomass of a leaf
    SLW::Float64 = 50 # Specific leaf weight (g/m2)
    LS::Float64  = 27   # Leaf shape parameter (length/width)
    maxwidthint::Float64=0.005 #max width of internode (m)
    max_width::Float64 =0.7249 #max width of leaf (m)
    shapeCoeff::Float64=0.2027 #shape coefficient of leaf
    specificPetiolelength::Float64 = 25 #petioles pecific length (m/g) 
    
    auxp::Float64=0.0 #auxin transport rate from shoot meristem to root meristem
    plastochron::Float64 =29#43 # Number of degree days between phytomer production
    phyllochron::Float64 =43#86
    delay::Float64 = phyllochron-plastochron
    leaf_expansion::Float64 = 220 # Number of degree day that a leaf expands
    potentialbiom_leaf::Float64=0 # g
    internode_expansion::Float64 = 182 # Number of degree day that a internode expands
    potentialbiom_internode::Float64=0 #g
    potentialbiom_sinternode::Float64=0 #
    phyllotaxis::Float64=137 #angle between two successive leaves
    lowerphyllotaxis::Float64=137
    leaf_angle::Float64=40 #angle between leaf and stem
    branch_angle::Float64=20.0 #angle for branches
    leaf_Curve::Float64=46
    max_leaf::Float64=11 #maximum leaf number
    max_shortint::Int64=2
    te_tuber::Float64=800
    potentialbiom_tuber::Float64=3
    leafnumber::Float64 =0 
    shortint::Int64=0
    availN::Float64=0.0
    intN::Float64=0.0
    tuberN::Float64=0.0
    #branching
    tiller::Int64=0
    maxtiller::Int64 = 60

#rootsystem parameters
    RLplasticity::Bool=true
    pinit::Bool=true #true: root primordia initiation
    pemerge::Bool=true #true: root primordia emergence
    ELauxin::Bool=true #true: auxin dependent root elongation
    lateral_angle::Float64 =140#lateral root angle 
    topangle::Float64 =rand(0:30)
    RB0::Float64 = 1e-8  # Initial biomass of a root
    root_biomass::Float64=0.0
    root_age::Int64=0
    root_ageD::Float64=0.0
    root_ageDprevious::Float64=0.0
    root_potentialbiom::Float64=1 #g
    root_expansion::Float64=800 # Number of degree day that root expands
    rank_root::Int64=0
    fN_root::Float64=0.01 #fraction of nitrogen in root (g/g)
    aroot_sink::Float64=0
    actbiomass_root::Float64=0
    actbiomass_rootperday::Float64=0
    leftrootbiomass::Float64=0
    val_root::Float64 =0 #axial root number
    lateralrootnumber::Float64=0
    rootlength::Float64=0
    arootlength::Float64=0
    lrootlength::Float64=0
    rootN::Float64=0.0
 #rootsegment parameters   
 width::Float64 = 0.002#initial diameter
 RDM::Float64=0.305 #ratio bwt mother and daughter root
 RTD::Float64=60#root tissue density (g/dm3)
 EL::Float64 =23 #root elongation rate m m-1 day-1
 ER::Float64=0.0056 #Emergence rate of root primordia per degree day    
 ELN::Float64=0 #root elongation rate in response to nitrogen
 IBD::Float64 = 0.001 #root segements length (m)
 arootseg::Int64=0 #axial root number
 max_root::Float64=0  #maximum axial root number 
 rootnumber::Float64 =0 #lateral root number 
 axialroot_angle::Float64=130 #axial root angle between upper stem and root 
 nodule_sink::Float64=0 #nodule sink
 nodulebiomass::Float64=0 #nodule biomass
 carcoeff::Float64=4.0 #carbon cost of nodule formation (C g/N g)

  #auxin 
  Naux::Bool=true #false: internal N is constant 
  exNaux::Bool=true #shootward auxin transport
  merauxin::Float64 =1 #auxin production rate in axial root meristem 
  decay::Float64 =0.43 #auxin decay ratio (per day)
  
 #fineroot parameters
  finerootdensity::Float64=8 #m fine root/m lateral root
  finerootD::Float64 =0.0001 #m fine root diameter

   #root uptake mechanism
   Imax::Float64 =20000 #umol/(m2*day) 
   Km::Float64 =50 #umol/l
   K2::Float64 =1.464  #umol/g
   Nup::Float64= 0.0 #g/plant
   Nfix::Float64=0.0 #g/plant
   accNfix::Float64=0.0 #g/plant
   Nupmol::Float64=0.0 #umol/plant
   potNup::Float64=0.0 #g/plant
   totalN::Float64=0 #g/plant
   species::String= ""
   monocot::Bool=true
end



function update_params!(p, new_values::Dict{String, Any})
    for (k, v) in new_values
        if hasfield(typeof(p), Symbol(k))
            field_type = fieldtype(typeof(p), Symbol(k))
            if field_type == Float64 && v isa Int
                setfield!(p, Symbol(k), Float64(v))
            else
                setfield!(p, Symbol(k), v)
            end
        else
            println("Warning: Field $k not found in $(typeof(p))")
        end
    end
    return p
end

# Function to load and update parameters from JSON file
function load_config(filename::String)
   
        params = JSON.parsefile(filename)
    

    if !isa(params, Dict)
        println("Error: Parsed JSON is not a dictionary")
        return nothing
    end

     rootbud_param = rootbud()
     rootNode_param = rootNode()
     emgrootbud_param = emgrootbud()
     lrootNode_param = lrootNode()
     shootrootNode_param = shootrootNode()
     BudNode_param = BudNode()
     rootBudNode_param = rootBudNode()
     meristem_param = meristem()
     rootmeristem_param = rootmeristem()
     lrootmeristem_param = lrootmeristem()
     Rootseg_param = Rootseg()
     Internode_param = Internode()
     Leaf_param = Leaf()
     Petiole_param = Petiole()
     tuber_param = tuber()
     Root_param = Root()
    spparams_param = spparams()
    
     # Update the instances with values from the JSON file
     update_params!(rootbud_param, params["rootbud"])
      update_params!(rootNode_param, params["rootNode"])
      update_params!(emgrootbud_param, params["emgrootbud"])
      update_params!(lrootNode_param, params["lrootNode"])
      update_params!(shootrootNode_param, params["shootrootNode"])
      update_params!(BudNode_param, params["BudNode"])
      update_params!(rootBudNode_param, params["rootBudNode"])
      update_params!(meristem_param, params["meristem"])
      update_params!(rootmeristem_param, params["rootmeristem"])
      update_params!(lrootmeristem_param, params["lrootmeristem"])
      update_params!(Rootseg_param, params["Rootseg"])
      update_params!(Internode_param, params["Internode"])
      update_params!(Leaf_param, params["Leaf"])
      update_params!(Petiole_param, params["Petiole"])
      update_params!(tuber_param, params["tuber"])
      update_params!(Root_param, params["Root"])
     update_params!(spparams_param, params["spparams"])
 
     return rootbud_param, rootNode_param, emgrootbud_param,lrootNode_param, shootrootNode_param, BudNode_param, 
     rootBudNode_param, meristem_param, rootmeristem_param, lrootmeristem_param,  
     Rootseg_param, Internode_param, Leaf_param, Petiole_param, tuber_param, Root_param,
     spparams_param
 end


end