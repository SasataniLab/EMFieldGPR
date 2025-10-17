--[[
    CoilのSTLファイルを出力する

    Output the STL file of coil.
]]

local save_stl_file_name = io.read()

--新しいプロジェクトの作成
app = cf.GetApplication()
project = app.Project

-- Created geometry: cylinder "Cylinder1"
properties = cf.Cylinder.GetDefaultProperties()
Cylinder1 = project.Geometry:AddCylinder({Base = {N = -0.0016}, Height = 0.0032, Radius = 0.12})

-- Created geometry: cylinder "Cylinder2"
properties = cf.Cylinder.GetDefaultProperties()
Cylinder2 = project.Geometry:AddCylinder({Base = {N = -0.0016}, Height = 0.0032, Radius = 0.09})

-- Created geometry: subtract "Subtract1"
Subtract1 = project.Geometry:Subtract(Cylinder1, Cylinder2)

-- Duplicated geometry: subtract Subtract1
Subtract2 = Subtract1:Duplicate()
Subtract3 = Subtract1:Duplicate()
Subtract4 = Subtract1:Duplicate()

-- Add translate transform
Translate2 = Subtract2.Transforms:AddTranslate(cf.Point(0,0,0),cf.Point(0, 0, -0.005))
Translate3 = Subtract3.Transforms:AddTranslate(cf.Point(0,0,0),cf.Point(0, 0, -0.010))
Translate4 = Subtract4.Transforms:AddTranslate(cf.Point(0,0,0),cf.Point(0, 0, -0.015))

-- Updating mesh parameters
MeshSettings = project.Mesher.Settings
properties = MeshSettings:GetProperties()
properties.Advanced.MinElementSize = 37.0511713132585
properties.Advanced.RefinementFactor = 62.4196350581785
properties.MeshSizeOption = cf.Enums.MeshSizeOptionEnum.Custom
properties.TriangleEdgeLength = 0.01
MeshSettings:SetProperties(properties)

-- Mesh the model
project.Mesher:Mesh()

--- Mesh export ---
project.Exporter.Mesh.ExportFileFormat = cf.Enums.ExportFileFormatEnum.STL

project.Exporter.Mesh.ExportMeshType = cf.Enums.ExportMeshTypeEnum.SimulationMesh

project.Exporter.Mesh.ExportOnlyBoundingFacesEnabled = true

project.Exporter.Mesh.ScaleToMetreEnabled = false

geometryTargets = { Subtract1, Subtract2, Subtract3, Subtract4 }

meshTargets = {  }

project.Exporter.Mesh:ExportParts(save_stl_file_name,geometryTargets,meshTargets)
