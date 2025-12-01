
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
import Parameters:@with_kw
import PlantGeomPrimitives:Mesh,slice!
using Revise
import IterTools

 includet("plant.jl")
    includet("Params.jl")
    includet("Growth.jl")
    includet("Nuptake.jl")
    includet("Nallocation.jl")
   
 
    import.Plant;
    import.speciestype;
    import.Growth;
    import.Nuptake;
    import.Nallocation;
    
     
    # intercrop::Bool=false #true: two crops
    leaching::Bool=true #true: N leaching happened
    raytrace::Bool=true #true: raytracing
    infinite::Bool=true #true: periodic boundary
    leafremoval::Bool=true #true: remove old leaves
    #splitroot::Bool=false#two root and reflection
    SAS::Bool=false #true: SAScoeficient for internode length
    harvest_date_potato=49#from emergence to harvest date
    harvest_date::Int64=0
    rep::Int64=10 #number of replications
    sowingdate=90 #sowing date
    DL=SkyDomes.day_length(52.0*π/180.0, SkyDomes.declination(sowingdate))*3600 #
    rowdistance_potato::Float64=0.1 #row distance for potato
    plantdistance_potato::Float64=0.025 #plant distance for potato
    nplant_potato::Int64=1 #number of plants in a row for potato
    nrows_potato::Int64=1 #number of rows for potato
    distance_between_crops::Float64=0 #planting row distance between two crops
    ref::Float64=0
    field_length::Float64=2.0 #field length in meters
    field_width::Float64=2.0 #field width in meters
    # if intercrop==true
    #     harvest_date=max(harvest_date_wheat, harvest_date_faba)
    # else
    #     if nplant_faba>0
    #         harvest_date=harvest_date_faba
    #     else
    #         harvest_date=harvest_date_wheat
    #     end
    # end 

    # x::Float64=10
    # y::Float64=10
    # origins=[Vec(i,j,0) for i in field_length/2:field_length:(x-0.5)*field_length, j in field_width/2:field_width:(y-0.5)*field_width];

    function temperature(dayofyear)
        #tav=9.12+15.72*sin(2*pi*(dayofyear-104)/365)
        tav=10.7+7.55*sin(2*pi*(dayofyear-111)/365)
        return tav
    end


    #light interception
    function create_soil()
        soil= Rectangle(length = field_length, width = field_width)
        rotatey!(soil, π/2) ## To put it in the XY plane
        VirtualPlantLab.translate!(soil, Vec(0.0, field_width/2, 0.0)) ## Corner at (0,0,0)
        return soil
    end

     function create_scene(field)    
    scene = Mesh(vec(field))
    #rt_settings= RTSettings(nx = 1, ny = 1,dx=field_length,dy=field_width)
    rt_settings= RTSettings(nx = 1, ny = 1,dx=10field_length,dy=10field_width,parallel = true)
    # Add a soil surface
    soil = create_soil() 
    soil_materials = VirtualPlantLab.Lambertian(τ = 0.0, ρ = 0.21)
    add!(scene, soil, materials = soil_materials)
    acc=accelerate(scene, settings = rt_settings, acceleration = BVH, rule = SAH{3}(5, 10))  # This creates the grid cloner
    return acc, rt_settings,scene
     end

    # function create_sky(;scene, lat = 42.0*π/180.0, DOY = sowingdate)
    # # Fraction of the day and day length
    # fs = collect(0.1:0.1:0.9)
    # dec = declination(DOY)
    # DL = day_length(lat, dec)*3600
    # # Compute solar irradiance
    # temp = [clear_sky(lat = lat, DOY = DOY, f = f) for f in fs] # W m2
    # Ig   = getindex.(temp, 1)
    # Idir = getindex.(temp, 2)
    # Idif = getindex.(temp, 3)
    # theta = getindex.(temp, 4)
    # phi = getindex.(temp, 5)
    # # Conversion factors to PAR for direct and diffuse irradiance
    # f_dir = waveband_conversion(Itype = :direct,  waveband = :PAR, mode = :power)
    # f_dif = waveband_conversion(Itype = :diffuse, waveband = :PAR, mode = :power)
    # # Actual irradiance per waveband
    # Idir_PAR = f_dir.*Idir
    # Idif_PAR = f_dif.*Idif
    # Itot = mean(Idif_PAR) + mean(Idir_PAR)
    # # Create the dome of diffuse light
    # dome = sky(scene,
    #                 Idir =0, ## No direct solar radiation
    #                 nrays_dif = 1_000, ## Total number of rays for diffuse solar radiation
    #                 Idif = sum(Idif_PAR)/10*DL, ## Daily Diffuse solar radiation
                    
    #                 sky_model = StandardSky, ## Angular distribution of solar radiation
    #                 dome_method = equal_solid_angles, ## Discretization of the sky dome
    #                 # Angles
    #                 ntheta = 9, ## Number of discretization steps in the zenith angle
    #                 #theta_dir = theta, ## Direction of the zenith angle
    #                 nphi = 12) ## Number of discretization steps in the azimuth angle
    #                 #phi_dir = phi) ## Direction of the azimuth angle
    # # Add direct sources for different times of the day
    # for i in eachindex(Idir_PAR)
    #     push!(dome, sky(scene, Idir = Idir_PAR[i]*DL/10, nrays_dir = 1_000, Idif = 0, theta_dir = theta[i], phi_dir = phi[i])[1])
    # end
    # return dome, Itot 
    # end

    # function create_raytracer(scene, sources)
    # settings = RTSettings(pkill = 0.9, maxiter = 4, nx = 5, ny = 5, parallel = true)
    # RayTracer(scene, sources, settings = settings);
    # end

    # function run_raytracer!(field; DOY)
    # scene   = create_scene(field)
    # sources, ref = create_sky(scene = scene, DOY = DOY)
    # rtobj   = create_raytracer(scene, sources)
    # trace!(rtobj)
    # return rtobj, ref
    # end

##photosynthesis leaf level

    @with_kw mutable struct Soilcell <: VirtualPlantLab.Node
        length::Float64 = 0.1#m
        width::Float64 = 0.1 #m
        height::Float64 = 0.1 #m
        x::Float64 =0.0
        y::Float64 = 0.0
        z::Float64 = 0.0
        i::Int64 = 0
        j::Int64 =0
        k::Int64 = 0
        N0::Float64=1000
        Nc::Float64=0 #umol/L
        Ni::Float64 = 0
        Nm::Float64=2*length*width*height*1000#umol/L
        potNup::Float64 =0 #umol/L
        rl::Float64= 0.0
        n::Int64=12
        material::VirtualPlantLab.Lambertian{1}=VirtualPlantLab.Lambertian(τ = 0.0, ρ = 0.21)
        #material::Vector{VirtualPlantLab.Lambertian{1}} = [VirtualPlantLab.Lambertian(τ = 0.0, ρ = 0.21) for _ in 1:n]
        #colors::Vector{RGBA}= [RGBA(Ni/N0, Ni/N0, 1-Ni/N0,0.1) for _ in 1:n]
    end

    function VirtualPlantLab.feed!(turtle::Turtle,ss::Soilcell, vars)
        t_pos=pos(turtle)
        t_head=head(turtle)
        center=t_pos .+ 0.5*ss.length*t_head
        ss.x=center[1]
        ss.y=center[2]
        ss.z=center[3]
        if turtle.message== "ray tracer"
           return nothing
        else
            HollowCube!(turtle,
            length = ss.length, 
            width = ss.width, 
            height = ss.height,
            colors=RGBA((ss.Ni)/1000, (ss.Ni)/1000, 1-(ss.Ni)/1000,0.1),
            materials=ss.material) # uncomment this cause error       
        end
        
    #Mesh!(turtle,sc,materials=ss.material,colors=copy(ss.colors))
    
    end    

    soil_graph = RA(-90.0) + T(Vec(0.0,0.0,0.0)) + # Moves into position
            #Tuple(RA(-90.0)+T(Vec(0.1,0.1,0.1))+Soilcell() for i in 1:1)             
    #Soil(length =2.0, width = 2.0)+
                Tuple(RA(-90.0) +T(Vec(Soilcell().length*j-Soilcell().length/2,Soilcell().width*i-Soilcell().width/2,0.1-0.1*k))+Soilcell(i=i,j=j,k=k) for j in 1:Int(ceil(field_length/Soilcell().length)) for i in 1:Int(ceil(field_width/Soilcell().width)) for k in 1:20)# Draws the soil tile
    soil = Graph(axiom = soil_graph);

    get_soils(soil)=apply(soil,Query(Soilcell))

  
    all_soils=get_soils(soil)
     function soil_Nc!(all_soils)
         for ss in all_soils
             if ss.z>-0.41&&ss.z<-0.21
             ss.Nc=2000.0
             else
                 ss.Nc=2000.0
             end
            ss.Ni=ss.Nc*ss.length*ss.width*ss.height*1000
         end
         end
        

    function Nleaching!(all_soils)
        for (cell1, cell2) in IterTools.product(all_soils, all_soils)
            distance = cell1.z - cell2.z
            if 0.099 < distance < 0.101 && round(cell1.x, digits=3) == round(cell2.x, digits=3) && round(cell1.y, digits=3) == round(cell2.y, digits=3)
                if cell1.Ni > cell1.Nm
                    transfer_amount = 0.05 * (cell1.Ni-cell1.Nm)
                    cell1.Ni -= transfer_amount
                    cell2.Ni += transfer_amount
                end
            end
        end
    end




    function reset_soil!(all_soils)
        for ss in all_soils
        ss.rl=0.0
        ss.Nc=0.0
        ss.potNup=0.0
        end
    end

    function reset_rootnumber!()
        rootnumber=Float64[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        halflrootnumber=Float64[0.0,0.0]
        return nothing
    end

    function reset_RLD!()
    RLD=Float64[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    halflrootlength=Float64[0.0,0.0]
    return nothing
    end

    function reset_interRLD!()
        interRLD=Float64[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        return nothing
        end
    RLD=Float64[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    interRLD=Float64[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    function calc_RLD!(field,plantdistance,rowdistance,field_length,field_width,nplant,nrows)
        halflrootlength=Float64[0,0]
        rl=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        arl=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        lrl=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        @threads for species in field
            vars=data(species)
        all_root=Plant.apply(species,Query(Plant.Rootseg))
        #all_lroot=Plant.apply(species,Query(Plant.lrootseg))
        
        for i in 1:20
        for r in all_root
            if r.order==0
            if -0.1*i<=r.z<=-0.1*(i-1)&&vars.row==1
                rl[i]+=1*r.IBD
                arl[i]+=r.IBD
            else
                rl[i]+=0
                arl[i]+=0
        end
    end
        
         if r.order==1
             if -0.1*i<=r.z<=-0.1*(i-1)&& vars.row==1
                 rl[i]+=vars.IBD
                 lrl[i]+=vars.IBD
             else
                 rl[i]+=0 
                 lrl[i]+=0
             end
            end
             end
        
    end
   
    end
    return rl, halflrootlength, arl, lrl
    end

    function calc_rootnumber!(field)
        rootnumber = zeros(Float64, 20*30)
        halflrootnumber=[0,0]
        @threads for species in field
        all_emgrootbud=Plant.apply(species,Query(Plant.emgrootbud))
        all_rootmeristem=Plant.apply(species,Query(Plant.rootmeristem))
        for j in 1:length(all_rootmeristem)
        for i in 1:20
        for rb in all_emgrootbud
         if -0.1*i<=rb.z<=-0.1*(i-1)&& rb.rank==j-1
            rootnumber[20*(j-1)+i]+=1
         end
        end
        end
        end
      
    for rb in all_emgrootbud

        if  rb.rank==1
            halflrootnumber[1]+=1
        elseif rb.rank==0
            halflrootnumber[2]+=1
        end
    end
    end
    return rootnumber, halflrootnumber
    end

    function write_root_data(; field, time::Int64, sim::Int64)
        # Initialize a thread-local array to store rows
        thread_local_data = Vector{Vector{Dict}}()
        # Collect data for each rootmeristem in the field
        @threads for sp in field
            local_data = Vector{Dict}()  # Thread-local data
            all_rootmeristem = Plant.apply(sp, Query(Plant.rootmeristem))
            for rm in all_rootmeristem
                row = Dict(
                    "time" => time,
                    "sim" => sim,
                    "plant_number" => :plant_number in fieldnames(typeof(data(sp))) ? getfield(data(sp), :plant_number) : missing,
                    "row" => :row in fieldnames(typeof(data(sp))) ? getfield(data(sp), :row) : missing,
                    "rank" => :rank in fieldnames(typeof(rm)) ? getfield(rm, :rank) : missing,
                    "ELN" => :ELN in fieldnames(typeof(rm)) ? getfield(rm, :ELN) : missing,
                    "totalauxin" => :totalauxin in fieldnames(typeof(rm)) ? getfield(rm, :totalauxin) : missing,
                    "shootauxin" => :shootauxin in fieldnames(typeof(rm)) ? getfield(rm, :shootauxin) : missing,
                    "merauxinN" => :merauxinN in fieldnames(typeof(rm)) ? getfield(rm, :merauxinN) : missing,
                    "totallength" => :totallength in fieldnames(typeof(rm)) ? getfield(rm, :totallength) : missing
                )
                push!(local_data, row)
            end
            # Store thread-local data
            push!(thread_local_data, local_data)
        end

        # Combine all thread-local data into a single vector
        combined_data = reduce(vcat, thread_local_data)
        # Convert to a DataFrame
        root_data = DataFrame(combined_data)
        return root_data
    end

    function write_lroot_data(; field, time::Int64, sim::Int64)
        # Initialize a thread-local array to store rows
        thread_local_data = Vector{Vector{Dict}}()
        # Collect data for each rootmeristem in the field
        @threads for sp in field
            local_data = Vector{Dict}()  # Thread-local data
            all_lrootmeristem = Plant.apply(sp, Query(Plant.lrootmeristem))
            for lrm in all_lrootmeristem
                row = Dict(
                    "time" => time,
                    "sim" => sim,
                    "plant_number" => :plant_number in fieldnames(typeof(data(sp))) ? getfield(data(sp), :plant_number) : missing,
                    "row" => :row in fieldnames(typeof(data(sp))) ? getfield(data(sp), :row) : missing,
                    "rank" => :rank in fieldnames(typeof(lrm)) ? getfield(lrm, :rank) : missing,
                    "ELN" => :ELN in fieldnames(typeof(lrm)) ? getfield(lrm, :ELN) : missing,
                    "totalauxin" => :totalauxin in fieldnames(typeof(lrm)) ? getfield(lrm, :totalauxin) : missing,
                    "shootauxin" => :shootauxin in fieldnames(typeof(lrm)) ? getfield(lrm, :shootauxin) : missing,
                    "merauxinN" => :merauxinN in fieldnames(typeof(lrm)) ? getfield(lrm, :merauxinN) : missing,
                    "totallength" => :totallength in fieldnames(typeof(lrm)) ? getfield(lrm, :totallength) : missing,
                    "total_sink" => :total_sink in fieldnames(typeof(lrm)) ? getfield(lrm, :total_sink) : missing)
                push!(local_data, row)
            end
            # Store thread-local data
            push!(thread_local_data, local_data)
        end

        # Combine all thread-local data into a single vector
        combined_data = reduce(vcat, thread_local_data)
        # Convert to a DataFrame
        lroot_data = DataFrame(combined_data)
        return lroot_data
    end


    function write_plant_data(;field,time::Int64, sim::Int64)
        plant_data = DataFrame()
        plant_data.time = [time for i in 1:length(field)]
        plant_data.sim=[sim for i in 1:length(field)]
        for n in fieldnames(Plant.speciestype.spparams)
            if (string(n) in ["species","plant_number","row","leafnumber","rootnumber","tiller","Nup","actbiomass","actbiomass_grain","actbiomass_leaf","actbiomass_stem","rs","Nc","accNfix","actbiomass_root","rootlength","lateralrootnumber","arootlength","lrootlength"])
                plant_data[:,n] = [getfield(data(p), n) for p in field]
            end
        end
            
    return plant_data
    end
     
    function write_field_data(;field, time::Int64, sim::Int64)
        # Build a DataFrame from the current field's plant data
        plant_df = DataFrame()
        plant_df.time = [time for _ in 1:length(field)]
        plant_df.sim = [sim for _ in 1:length(field)]
        for n in fieldnames(Plant.speciestype.spparams)
            if (string(n) in ["species","plant_number","row","leafnumber","rootnumber","tiller","Nup","actbiomass","actbiomass_grain","actbiomass_leaf","actbiomass_stem","rs","Nc","accNfix","actbiomass_root","rootlength","lateralrootnumber","arootlength","lrootlength"])
                plant_df[:,n] = [getfield(data(p), n) for p in field]
            end
        end
        # Group and aggregate on the current simulation's plant data
        field_data = combine(groupby(plant_df, [:sim, :time]), 
            :actbiomass => sum, 
            :rootlength => sum, 
            :actbiomass_grain => sum, 
            :actbiomass_leaf => sum, 
            :actbiomass_stem => sum, 
            :actbiomass_root => sum, 
            :Nup => sum)
        return field_data
    end

    function write_soil_data(all_soils; time::Int64, sim::Int64)
        # Initialize a thread-local array to store rows
        thread_local_data = Vector{Vector{Dict}}()
        # Collect data for each soil cell
        #@threads for ss in all_soils
        for ss in all_soils
            local_data = Vector{Dict}()  # Thread-local data
            row = Dict(
                "time" => time,
                "sim" => sim,
                "x" => ss.x,
                "y" => ss.y,
                "z" => ss.z,
                "Ni" => ss.Ni,
                "Nc" => ss.Nc,
                "rl" => ss.rl,
                "potNup" => ss.potNup
            )
            push!(local_data, row)
            # Store thread-local data
            push!(thread_local_data, local_data)
        end
        # Combine all thread-local data into a single vector
        combined_data = reduce(vcat, thread_local_data)
        # Convert to a DataFrame
        soil_data = DataFrame(combined_data)
        return soil_data
    end

    function render_field(field, soil,filename::String)
    #scene = Scene(vec(field), message = "ray tracer")
    scene = Mesh(vec(field))
        Soil=Mesh([soil])
    
        scene = Mesh([scene,Soil])
        
        # merges the two scenes
        f = render(scene, wireframe = false, normals = false)
        #GLMakie.save(filename, f, px_per_unit = 1)  
        return f
    end


    RLDtotal_potato= DataFrame(layer=1:20)
    lrootlength= DataFrame(layer=1:20)
    arootlength= DataFrame(layer=1:20)
    interRLDtotal= DataFrame(layer=1:20)
    plant=DataFrame()
    Field=DataFrame()
    root_potato=DataFrame()
    lroot_potato=DataFrame()
    soil_data=DataFrame()
    Rootnumber_potato = DataFrame(layer = repeat(1:20, 30))
    halfRootnumber_potato=DataFrame(part=["1","0"])
    halflrootlength_potato=DataFrame(part=["1","0"])

        for j in 1:rep
            
            orientations_potato =[rand()*360.0 for i = plantdistance_potato/2:plantdistance_potato:nplant_potato*plantdistance_potato-plantdistance_potato/2, j = rowdistance_potato:rowdistance_potato:rowdistance_potato*nrows_potato]
            origins_potato =[Vec(i,j,0) for i = plantdistance_potato/2:plantdistance_potato:nplant_potato*plantdistance_potato-plantdistance_potato/2, j = rowdistance_potato/2:rowdistance_potato:rowdistance_potato*nrows_potato-rowdistance_potato/2];
            fieldpotato= [Plant.create_plant(origins_potato[i],i,Int(ceil(i/nplant_potato)),orientations_potato[i],Plant.potato_param) for i in 1:Int(nplant_potato*nrows_potato)];
            field = fieldpotato
            render_field(field, soil,"output$j.0.png")
            soil_Nc!(all_soils)
        println("Rep $j")
        for i in 1:harvest_date
        println("Day $i")
        dayofyear=i+sowingdate
        println("Day_potato $i")
         if raytrace==true
            Growth.canopy_photosynthesis!(field,dayofyear)
        end
        create_scene(field)
        Plant.daily_step!(field,soil,dayofyear,DL,i)

        if leaching==true
            Nleaching!(all_soils)
        end

    append!(plant, write_plant_data(field=field, time=i+sowingdate, sim=j))
    append!(Field, write_field_data(field=field, time=i+sowingdate, sim=j))
    #root_wheat=vcat(root_wheat, write_root_data(field=field, time=i+sowingdate, sim=j))  
    #lroot_wheat=vcat(lroot_wheat, write_lroot_data(field=field, time=i+sowingdate, sim=j))
    append!(soil_data, write_soil_data(all_soils, time=i+sowingdate, sim=j))      
        #write_graph_data(field = new_field[1], plants = new_field[2], time = time, location = output_location, output_name = output_name_graphs)
       Growth.reset_biomass!(field)
        #calc_RLD!(all_aroots,all_lroots)
        if mod(i,harvest_date)==0
            display(render_field(field, soil,"output$j.$i.png"))
        #export_makie_graph(field, "output$j.$i.png")   
        end 
    end 

    calc_rootnumber!(field)
    calc_RLD!(fieldpotato,plantdistance_potato,rowdistance_potato,field_length,field_width,nplant_potato,nrows_potato)   
    # RLDtotal_wheat = hcat(RLDtotal_wheat,calc_RLD!(fieldwheat,plantdistance_wheat,rowdistance_wheat,field_length,field_width,nplant_wheat,nrows_wheat)[1], makeunique=true)
    col_name = Symbol("RLD_rep_$j")
    RLDtotal_potato = hcat(RLDtotal_potato, DataFrame(col_name => calc_RLD!(fieldpotato,plantdistance_potato,rowdistance_potato,field_length,field_width,nplant_potato,nrows_potato)[1]), makeunique=true)    
        #halflrootlength_wheat=hcat(halflrootlength_wheat,calc_RLD!(fieldwheat,plantdistance_wheat,rowdistance_wheat,field_length,field_width,nplant_wheat,nrows_wheat)[2],makeunique=true)
        #arootlength=hcat(arootlength,calc_RLD!(fieldwheat,plantdistance_wheat,rowdistance_wheat,field_length,field_width,nplant_wheat,nrows_wheat)[3],makeunique=true)
        #lrootlength=hcat(lrootlength,calc_RLD!(fieldwheat,plantdistance_wheat,rowdistance_wheat,field_length,field_width,nplant_wheat,nrows_wheat)[4],makeunique=true)
        reset_soil!(get_soils(soil))
        reset_RLD!()
        reset_interRLD!()
        
        #Rootnumber_wheat=hcat(Rootnumber_wheat,calc_rootnumber!(fieldwheat)[1],makeunique=true)
        #halfRootnumber_wheat=hcat(halfRootnumber_wheat,calc_rootnumber!(fieldwheat)[2],makeunique=true)
        reset_rootnumber!()
        
    end

    CSV.write("C:/Users/angel021/VSCODE/potato/phfield.csv",Field)

    CSV.write("C:/Users/angel021/VSCODE/potato/phplant.csv",plant)
    CSV.write("C:/Users/angel021/VSCODE/potato/phroot.csv",root_potato)
    CSV.write("C:/Users/angel021/VSCODE/potato/phRootnumber.csv",Rootnumber_potato)
    CSV.write("C:/Users/angel021/VSCODE/potato/phlroot.csv",lroot_potato)
    CSV.write("C:/Users/angel021/VSCODE/potato/phsoil.csv",soil_data)


    CSV.write("C:/Users/angel021/VSCODE/potato/phhalfRootlength.csv",halflrootlength_potato)
    CSV.write("C:/Users/angel021/VSCODE/potato/phhalfRootnumber.csv",halfRootnumber_potato)
    CSV.write("C:/Users/angel021/VSCODE/potato/phRLD.csv", RLDtotal_potato)
    CSV.write("C:/Users/angel021/VSCODE/potato/pharootlength.csv", arootlength)
    CSV.write("C:/Users/angel021/VSCODE/potato/phlrootlength.csv", lrootlength)
    #1. lateral root angle: plant.jl line 188
    #2. lateral root emergence: plant.jl line 508-515 (auxin)
    #3. soil N distribution: Field.jl Line 238-249
    #4. lateral root initiation: line 341 (auxin)
    #5. soil N change the auxin distrubution: plant.jl line 1393-1401,1409-1423
    #6. plant N change the shoot auxin distribution: plant.jl line 1041-1045, 1000-1003
end