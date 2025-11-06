function foo(x)
    x^2
end
foo(2.0)
foo(3.0)
foo2(y)=y+2
foo2(5)
foo3 = x -> x^2
foo3(2)

# Must use '->' for making function with this format
foo4 = begin
    x -> x^2
end
foo4(2)

# using '=' for making function causes error

#'=' means equals/assign value to a variable
#'->' means action, defining the input and output
foo5=begin
    x = x^2
end 

function bar(x,f)
    f(x)
end
bar(2.0, x -> x^2)
bar(2.0, x = x^2)

# Immutable point
struct Point
    x
    y
    z
end
p = Point(0.0, 0.0, 0.0)
# Immutable means once created, the fields of an object cannot be changed
p.x = 1.0

# Mutable point
mutable struct mpoint
    x 
    y 
    z 
end
mp = mpoint(0.0, 0.0, 0.0)
mp.x = 1.0
mp.y = 2.0
mp.z = 4.0
mp

#check type of object
typeof(mp)
typeof(p)

# Return the name of all the fields it contains
fieldnames(Point)

# For performance reasons, the type of each field should be annotated with the type definition
struct pPoint
    x::Float64
    y::Float64
    z::Float64
end
pPoint(1.0,2.0,3.0)
pPoint(2, 3, 4)
a=-4
b=2
c = a+b
pPoint(a,b,c)

# Methods = functions that specialised for different types

function dist(p1::pPoint, p2::pPoint)
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
    sqrt(dx^2 + dy^2 + dz^2)
end

p1 = pPoint(1, 0, 0)
p2 = pPoint(0,1,0)
dist(p1, p2)

# The 'dist' method is not working with mpoint as p1 and p2 is not defined as 'mpoint'
mp1 = mpoint(1,0,0)
mp2 = mpoint(0,1,0)
dist(mp1, mp2)

# That means we need to define new 'dist' for 'mpoint' as arguments, or use inheritance.
# Inheritance is used for abstract types 
abstract type Vec3 end
function dist_vec(p1::Vec3, p2::Vec3)
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
    sqrt(dx^2 + dy^2 + dz^2)
end
struct Point2 <: Vec3
    x::Float64
    y::Float64
    z::Float64
end
mutable struct mPoint2 <: Vec3
    x::Float64
    y::Float64
    z::Float64
end
struct Point3 <: Vec3
    x::Float64
    y::Float64
end

# The methods now works with Point2 and mPoint2
p1 = Point2(1, 0, 0)
p2 = Point2(0, 1, 0)
dist_vec(p1, p2)

# The method will not try to run with Point3 but will raise an error since Point3 doesn't have the field z
p3 = Point3(1,0)
dist_vec(p1, p3)

# Optional and keyword arguments
opfoo(a, b::Int = 0) = a+ b
opfoo(1)
opfoo(1,1)

kwfoo(a; b::Int = 0) = a+b
kwfoo(1)
kwfoo(1, b =1)

# It has to follow the sequence
kwfoo(b=1, a=1)

module Mod

export fooz

fooz(x) = abs(x)

struct bar
    data    
end

end

using .Mod

fooz(-2)

# Unexported names can still be retrieved, but must be qualified by the module name 
b = Mod.fooz(-1)

module Funs
    export manhattan
    function manhattan(p1, p2)
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        abs(dx + dy + dz)
    end
end

using .Funs
manhattan(p1, p2)
manhattan(p1,p3)

methods(manhattan)

#Macro: 

Base.@kwdef struct kwPoint
    x::Float64
    y::Float64
    z::Float64
end
kwPoint()
kwPoint(1,1,1)
kwPoint(y=1)

x=[1,2,3]
y = x.^2

abs.(y)

abs.(y) .+ x.^3
@. abs(y) + x^3

min.(x,y)
max.(x,y)

function add_squares(x)
    out = 0
    for i in eachindex(x)
        out += x[i]^2
    end
    return out
end

add_squares(collect(1:1000))
add_squares(collect(1:1000.0))

using BenchmarkTools
v1 = collect(1:1000)
v2 = collect(1:1000.0)
@btime add_squares($v1)
@btime add_squares($v2)

@code_warntype add_squares(v1)
@code_warntype add_squares(v2)

function add_squares_new(x)
    out = zero(eltype(x)) # initialize out with the correct type with value of zero
    for i in eachindex(x)
        out += x[i]^2
    end
    return out
end

@code_warntype add_squares_new(v1)
@code_warntype add_squares_new(v2)

@btime add_squares_new($v1)
@btime add_squares_new($v2)
