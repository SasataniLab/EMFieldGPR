--[[
    STLファイルを出力する

    Output the STL file.
]]

local load_pattern_file_name = io.read()
local save_stl_file_name = io.read()

--新しいプロジェクトの作成
app = cf.GetApplication()
project = app:NewProject()

--単位の設定
project.ModelAttributes.Unit = cf.Enums.ModelUnitEnum.Metres

-------------------------------------
--初期設定
-------------------------------------
this_config = project.SolutionConfigurations[1]
--media(材質)の設定
Copper = project.Media.Library:AddToModel("Copper")
-- Ferriteの設定
Ferrite = project.Media:AddDielectric()
    -- Modify the dielectric to use frequency independent magnetic modelling
Ferrite.MagneticModelling.DefinitionMethod =
    cf.Enums.MediumMagneticDefinitionMethodEnum.FrequencyIndependent
Ferrite.MagneticModelling.RelativePermeability = 3300
Ferrite.MagneticModelling.LossTangent = 1/1000
Ferrite.MassDensity = 4900

-- Change the colour to Black
Ferrite.Colour = "#000000"

--extents
-- app.Project.ModelAttributes.ExtentsExponent = 2

--double precision に設定
project.SolutionSettings.SolverSettings.GeneralSettings.DataStoragePrecision = cf.Enums.PrecisionSettingsEnum.Double

--portを準備
all_port={}

rad_st = 0.09
rad_end = 0.12
turns = 7
left0 = false
left1 = true

frequency = 85e3
pi = 3.14159265358979323846

--Ferriteシールドの生成
ferrite_pattern = {}

file = io.open(load_pattern_file_name, "r")
cnt = 1
for line in file:lines("l") do
    for word in string.gmatch(line, "%S+") do
        ferrite_pattern[cnt] = tonumber(word)
        cnt = cnt + 1
    end
end
file:close()

Cylinder1 = project.Geometry:AddCylinder({Base = {N = -0.02 - ferrite_pattern[5]}, Height = ferrite_pattern[5], Radius = ferrite_pattern[1]})
if not (ferrite_pattern[6] == 0) and not (ferrite_pattern[7] == 0) and not (ferrite_pattern[2] == ferrite_pattern[3]) then
    Cylinder2 = project.Geometry:AddCylinder({Base = {N = -0.02}, Height = ferrite_pattern[6], Radius = ferrite_pattern[2]})
    Cylinder3 = project.Geometry:AddCylinder({Base = {N = -0.02}, Height = ferrite_pattern[6], Radius = ferrite_pattern[3]})
    Subtract1 = project.Geometry:Subtract(Cylinder2, Cylinder3)
    Cylinder4 = project.Geometry:AddCylinder({Base = {N = -0.02}, Height = ferrite_pattern[7], Radius = ferrite_pattern[4]})
    Union2 = project.Geometry:Union({Cylinder1, Subtract1, Cylinder4})
elseif not (ferrite_pattern[6] == 0) and ferrite_pattern[7] == 0 and not (ferrite_pattern[2] == ferrite_pattern[3]) then
    Cylinder2 = project.Geometry:AddCylinder({Base = {N = -0.02}, Height = ferrite_pattern[6], Radius = ferrite_pattern[2]})
    Cylinder3 = project.Geometry:AddCylinder({Base = {N = -0.02}, Height = ferrite_pattern[6], Radius = ferrite_pattern[3]})
    Subtract1 = project.Geometry:Subtract(Cylinder2, Cylinder3)
    Union2 = project.Geometry:Union({Cylinder1, Subtract1})
elseif ferrite_pattern[6] == 0 and not (ferrite_pattern[7] == 0) then
    Cylinder4 = project.Geometry:AddCylinder({Base = {N = -0.02}, Height = ferrite_pattern[7], Radius = ferrite_pattern[4]})
    Union2 = project.Geometry:Union({Cylinder1, Cylinder4})
else
    Union2 = Cylinder1
end

Shield1 = project.Geometry:Simplify(Union2)
for key, value in pairs(Shield1.Regions) do
    value.Medium = Ferrite
    value.SolutionMethod = cf.Enums.RegionSolutionMethodEnum.VEP
end

-- メッシュ作成
project.Mesher.Settings.WireRadius = 0.0016
properties = {}
properties.MeshSizeOption = "Custom"
properties.TriangleEdgeLength = 0.01
properties.WireSegmentLength = 0.005
properties.TetrahedronEdgeLength = 0.01
project.Mesher.Settings:SetProperties(properties)

project.Mesher:Mesh()

--- Mesh export ---
project.Exporter.Mesh.ExportFileFormat = cf.Enums.ExportFileFormatEnum.STL

project.Exporter.Mesh.ExportMeshType = cf.Enums.ExportMeshTypeEnum.SimulationMesh

project.Exporter.Mesh.ExportOnlyBoundingFacesEnabled = true

project.Exporter.Mesh.ScaleToMetreEnabled = false

geometryTargets = { Shield1 }

meshTargets = {  }

project.Exporter.Mesh:ExportParts(save_stl_file_name,geometryTargets,meshTargets)