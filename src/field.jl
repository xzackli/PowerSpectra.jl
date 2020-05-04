

struct Field{T}
    name::String
    maskT::Healpix.Map{T}
    σTT::Healpix.Map{T}
end

struct PolarizedField{T}
    name::String
    maskT::Healpix.Map{T}
    maskP::Healpix.PolarizedMap{T}
    σTT::Healpix.Map{T}
    σPP::Healpix.PolarizedMap{T}
end

function Base.show(io::IO, ::MIME"text/plain", x::Field{T}) where T
    println("Field " * x.name, ": ", typeof(x.maskT), " ", size(x.maskT.nside))
    println("maskT [", ["$(x_), " for x_ in x.maskT.pixels[1:3]]..., "...]")
    println("σII   [", ["$(x_), " for x_ in x.maskT.pixels[1:3]]..., "...]")
end
