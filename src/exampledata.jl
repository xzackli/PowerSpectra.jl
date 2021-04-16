# example data (udgrade to nside 256 of the Planck 2018 maps)

function util_planck256_dir()
    rootpath = artifact"plancklowres"
    return rootpath
end


function util_planck256_map(freq, split, col, type::Type=Float64)
    rootpath = artifact"plancklowres"
    fname = "nside256_HFI_SkyMap_$(freq)_2048_R3.01_halfmission-$(split).fits"
    return readMapFromFITS(joinpath(rootpath, "plancklowres", "maps", fname), col, type)
end

function util_planck256_polmap(freq, split, type::Type=Float64)
    return PolarizedMap(
        util_planck256_map(freq, split, 1, type), 
        util_planck256_map(freq, split, 2, type), 
        util_planck256_map(freq, split, 3, type))
end

function util_planck256_maskT(freq, split, type::Type=Float64)
    rootpath = artifact"plancklowres"
    fname = "nside256_COM_Mask_Likelihood-temperature-$(freq)-hm$(split)_2048_R3.00.fits"
    return readMapFromFITS(joinpath(rootpath, "plancklowres", "masks", fname), 1, type)
end

function util_planck256_maskP(freq, split, type::Type=Float64)
    rootpath = artifact"plancklowres"
    fname = "nside256_COM_Mask_Likelihood-polarization-$(freq)-hm$(split)_2048_R3.00.fits"
    return readMapFromFITS(joinpath(rootpath, "plancklowres", "masks", fname), 1, type)
end
