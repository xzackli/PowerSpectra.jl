

struct Field{T}
    name::String
    maskT::Map{T}
    σTT::Map{T}
end

struct PolarizedField{T}
    name::String
    maskT::Map{T}
    maskP::PolarizedMap{T}
    σTT::Map{T}
    σPP::PolarizedMap{T}
end

function Base.show(io::IO, ::MIME"text/plain", x::Field{T}) where T
    println("Field " * x.name, ": ", typeof(x.maskT), " ", size(x.maskT.nside))
    println("maskT [", ["$(x_), " for x_ in x.maskT.pixels[1:3]]..., "...]")
    println("σII   [", ["$(x_), " for x_ in x.maskT.pixels[1:3]]..., "...]")
end
