--[[
    SParameterを求める
    コマンドライン引数としてフェライトシールドを特定するためのファイル名をもらっている

    Determine the SParameter.
    Receiving a filename as a command line argument to identify the ferrite shield.
]]


local save_file_name = io.read()
local load_pattern_file_name = io.read()

--新しいプロジェクトの作成
app = cf.GetApplication()
project = app:NewProject()

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
y_move = ferrite_pattern[8]
z_move = ferrite_pattern[9]


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

----------
--スパイラルコイル(送電器)
----------
--:AddEllipticArcスタートポイント，半径，半径，スタートの角度，終わりの角度
Spiral1 = project.Geometry:AddHelix(cf.Point(0,0,0),rad_st,rad_end,0,turns,left0)
Spiral1.Wires[1].CoreMedium =Copper
Spiral2 = project.Geometry:AddHelix(cf.Point(0,0,0.005),rad_st,rad_end,0,turns,left1)
Spiral2.Wires[1].CoreMedium =Copper
Spiral3 = project.Geometry:AddHelix(cf.Point(0,0,0.010),rad_st,rad_end,0,turns,left0)
Spiral3.Wires[1].CoreMedium =Copper
Spiral4 = project.Geometry:AddHelix(cf.Point(0,0,0.015),rad_st,rad_end,0,turns,left1)
Spiral4.Wires[1].CoreMedium =Copper
--:ショートさせるための線を作成
Line1 = project.Geometry:AddLine(cf.Point(rad_st,0,0),cf.Point(rad_st,0,0.005))
Line1.Wires[1].CoreMedium = Copper
Line2 = project.Geometry:AddLine(cf.Point(rad_end,0,0.005),cf.Point(rad_end,0,0.010))
Line2.Wires[1].CoreMedium = Copper
port_sp = project.Ports:AddWirePort(Line2.Wires[1])
port_sp.Location = cf.Enums.WirePortLocationEnum.Middle
all_port[1] = port_sp
Line3 = project.Geometry:AddLine(cf.Point(rad_st,0,0.010),cf.Point(rad_st,0,0.015))
Line3.Wires[1].CoreMedium = Copper
PolyLine = project.Geometry:AddPolyline({cf.Point(rad_end,0,0), cf.Point(rad_end + 5e-3,0,0),cf.Point(rad_end + 5e-3,0,0.015),cf.Point(rad_end,0,0.015)})
PolyLine.Wires[1].CoreMedium =Copper
PolyLine.Wires[2].CoreMedium =Copper
PolyLine.Wires[3].CoreMedium =Copper
--portを取り付ける
port_sp = project.Ports:AddWirePort(PolyLine.Wires[2])
port_sp.Location = cf.Enums.WirePortLocationEnum.Middle
all_port[2] = port_sp
--Unionを作成
Coil1 = project.Geometry:Union({Spiral1, Spiral2, Spiral3, Spiral4, Line1, Line2, Line3, PolyLine})
Coil1_rotate = Coil1.Transforms:AddRotate(cf.Point(0,0,0), cf.Point(1.0,0,0), 180)

Coil2 = Coil1:Duplicate()
Coil2_rotate = Coil2.Transforms:AddRotate(cf.Point(0,0,0), cf.Point(1.0,0,0), 180)
Coil2_transform = Coil2.Transforms:AddTranslate(cf.Point(0,0,0),cf.Point(0, y_move, 0.05 + z_move))
port_sp = project.Ports:AddWirePort(Coil2.Wires[2])
port_sp.Location = cf.Enums.WirePortLocationEnum.Middle
all_port[3] = port_sp
port_sp = project.Ports:AddWirePort(Coil2.Wires[6])
port_sp.Location = cf.Enums.WirePortLocationEnum.Middle
all_port[4] = port_sp

--Ferriteシールドの生成

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

Shield2 = Shield1:Duplicate()
Shield2_rotate = Shield2.Transforms:AddRotate(cf.Point(0,0,0), cf.Point(1.0,0,0), 180)
Shield2_transform = Shield2.Transforms:AddTranslate(cf.Point(0,0,0),cf.Point(0, y_move, 0.05 + z_move))



-------------------------------------
--simulation 設定
-------------------------------------
-- 周波数の設定
this_config.Frequency.Start = frequency

-- -- port役割を与える
voltage_source1 = this_config.Sources:AddVoltageSource(all_port[2])
voltage_source1.Magnitude = 1
voltage_source1.Phase = 0
voltage_source1.Impedance = 50
this_config.Included = false


-- Sパラメータの設定
solutionConfigurations = project.SolutionConfigurations
SParameterConfiguration = solutionConfigurations:AddMultiportSParameter({all_port[2],all_port[3]})
SParametersRequest = SParameterConfiguration.SParameter
SParametersRequest.TouchstoneExportEnabled = true
-- --SParametersRequest:ExportData(miso_4by4,6.78e6,Scattring,RI)

-- メッシュ作成
project.Mesher.Settings.WireRadius = 0.0016
properties = {}
properties.MeshSizeOption = "Custom"
--properties.TriangleEdgeLength = "2"
properties.TriangleEdgeLength = 0.01
properties.WireSegmentLength = 0.005
properties.TetrahedronEdgeLength = 0.02
project.Mesher.Settings:SetProperties(properties)

project.Mesher:Mesh()


-- 最後に視点を全体にする
app.Views[1]:ZoomToExtents()

app:SaveAs(save_file_name)

-- -- FEKOの実行
project.Launcher:RunFEKO()