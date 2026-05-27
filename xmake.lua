add_rules("mode.debug", "mode.release")

add_requires("cuda")

target("deep-learning-c")
    set_kind("binary")
    set_languages("c11")
    add_includedirs("src")
    add_files("src/**.c")
    
    if is_plat("linux") then
        add_defines("_GNU_SOURCE")
    end

target("deep-learning-c_cuda")
    set_kind("binary")
    set_languages("c11", "cu")
    add_includedirs("src")
    add_files("src/**.c")
    local cu_files = os.files("src/**.cu")
    if #cu_files > 0 then
        add_files("src/**.cu")
    end
    add_packages("cuda")
    add_defines("USE_CUDA")
    add_links("c")

    if is_plat("linux") then
        add_defines("_GNU_SOURCE")
    end
