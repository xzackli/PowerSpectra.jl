# example data (udgrade to nside 256 of the Planck 2018 maps)

function planck256_mapdir()
    rootpath = artifact"planck_pr3_lowres_maps"
    return rootpath
end

function planck256_beamdir()
    rootpath = artifact"planck_pr3_beams"
    return rootpath
end

function planck_FITS_name_to_col(name_)
    name = String(name_)
    (name == "I_STOKES") && return 1
    (name == "Q_STOKES") && return 2
    (name == "U_STOKES") && return 3
    (name == "HITS")     && return 4
    (name == "II_COV")   && return 5
    (name == "IQ_COV")   && return 6
    (name == "IU_COV")   && return 7
    (name == "QQ_COV")   && return 8
    (name == "QU_COV")   && return 9
    (name == "UU_COV")   && return 10
end

"""
    planck256_map(freq, split, col, type::Type=Float64) -> Map{T, RingOrder}

Returns a Planck 2018 half-mission frequency map downgraded to nside 256 in KCMB units.
FITS file column numbers are 

1. I_STOKES
2. Q_STOKES
3. U_STOKES
4. HITS
5. II_COV
6. IQ_COV
7. IU_COV
8. QQ_COV
9. QU_COV
10. UU_COV

# Arguments:
- `freq::String`: Planck frequency ∈ {"100", "143", "217"}
- `split::String`: half mission split ∈ {"hm1", "hm2"}
- `col::Int`: FITS file column. Either a number, String, or Symbol above. 

# Returns: 
- `Map{T, RingOrder}`: the map
"""
function planck256_map(freq::String, split, col, type::Type=Float64)
    if typeof(col) != Int
        col = planck_FITS_name_to_col(col)
    end
    @assert split[1:2] == "hm"
    rootpath = planck256_mapdir()
    fname = "nside256_HFI_SkyMap_$(freq)_2048_R3.01_halfmission-$(split[3]).fits"
    return nest2ring(readMapFromFITS(joinpath(rootpath, "plancklowres", fname), col, type))
end

function planck256_polmap(freq, split, type::Type=Float64)
    return PolarizedMap(
        planck256_map(freq, split, 1, type), 
        planck256_map(freq, split, 2, type), 
        planck256_map(freq, split, 3, type))
end

function planck256_maskT(freq, split, type::Type=Float64)
    rootpath = planck256_mapdir()
    fname = "nside256_COM_Mask_Likelihood-temperature-$(freq)-$(split)_2048_R3.00.fits"
    return readMapFromFITS(joinpath(rootpath, "plancklowres", fname), 1, type)
end

function planck256_maskP(freq, split, type::Type=Float64)
    rootpath = planck256_mapdir()
    fname = "nside256_COM_Mask_Likelihood-polarization-$(freq)-$(split)_2048_R3.00.fits"
    return readMapFromFITS(joinpath(rootpath, "plancklowres", fname), 1, type)
end


function planck_beam_bl(T::Type, freq1, split1, freq2, split2, spec1_, spec2_; 
                          lmax=4000)
    spec1 = String(spec1_)
    spec2 = String(spec2_)

    rootpath = planck256_beamdir()

    if parse(Int, freq1) > parse(Int, freq2)
        freq1, freq2 = freq2, freq1
        split1, split2 = split2, split1
    end
    if (parse(Int, freq1) == parse(Int, freq2)) && ((split1 == "hm2") && (split1 == "hm1"))
        split1, split2 = split2, split1
    end

    fname = "Wl_R3.01_plikmask_$(freq1)$(split1)x$(freq2)$(split2).fits"
    f = FITS(joinpath(rootpath, "planckbeams_plikmask", fname))
    bl = convert(Vector{T}, read(f[spec1], "$(spec1)_2_$(spec2)")[:,1])
    if lmax < 4000
        bl = bl[1:lmax+1]
    else
        bl = vcat(bl, last(bl) * ones(T, lmax - 4000))
    end
    return SpectralVector(bl)
end
planck_beam_bl(T::Type, freq1, split1, freq2, split2, spec1; kwargs...) = 
    lanck_beam_bl(T, freq1, split1, freq2, split2, spec1, spec1; kwargs...)
planck_beam_bl(freq1::String, split1, freq2, split2, spec1, spec2; kwargs...) = 
    planck_beam_bl(Float64, freq1, split1, freq2, split2, spec1, spec2; kwargs...)


function planck_theory_Dl()
    rootpath = artifact"planck_pr3_theory"
    fname = "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
    table = read_commented_header(joinpath(rootpath, fname))
    
    syms = (:TT, :TE, :EE, :BB, :PP)
    return NamedTuple{syms}(map(
        XY -> SpectralVector([0.0, 0.0, table[!, XY]...]), syms))
end
