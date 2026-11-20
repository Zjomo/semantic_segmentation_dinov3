<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="20008000">
	<Property Name="NI.LV.All.SourceOnly" Type="Bool">false</Property>
	<Property Name="NI.Project.Description" Type="Str">JY光谱仪温度测量与Mapping功能扩展项目
基于拉曼位移-温度校准曲线法的温度测量系统
支持面扫描（Mapping）数据采集和CSV格式导出

创建日期：2026-05-14
版本：1.0</Property>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="主程序" Type="Folder">
			<Item Name="JY_Temperature_Mapping.vi" Type="VI" URL="JY_Temperature_Mapping.vi"/>
		</Item>
		<Item Name="子程序" Type="Folder">
			<Item Name="Read_Calibration_File.vi" Type="VI" URL="Read_Calibration_File.vi"/>
			<Item Name="Find_Raman_Peak.vi" Type="VI" URL="Find_Raman_Peak.vi"/>
			<Item Name="Calculate_Temperature.vi" Type="VI" URL="Calculate_Temperature.vi"/>
			<Item Name="Generate_Scan_Path.vi" Type="VI" URL="Generate_Scan_Path.vi"/>
			<Item Name="Export_to_CSV.vi" Type="VI" URL="Export_to_CSV.vi"/>
			<Item Name="Plot_Temperature_Map.vi" Type="VI" URL="Plot_Temperature_Map.vi"/>
		</Item>
		<Item Name="配置和数据" Type="Folder">
			<Item Name="Calibration" Type="Folder">
				<Item Name="Raman_Temperature_Calibration.txt" Type="Document" URL="Calibration/Raman_Temperature_Calibration.txt"/>
			</Item>
			<Item Name="Results" Type="Folder">
				<Item Name="Temperature_Data" Type="Folder">
				</Item>
			</Item>
		</Item>
		<Item Name="原有程序（备份参考）" Type="Folder">
			<Item Name="JY.vi" Type="VI" URL="JY.vi"/>
			<Item Name="JY Mono.vi" Type="VI" URL="JY Mono.vi"/>
			<Item Name="JYSp.vi" Type="VI" URL="JYSp.vi"/>
			<Item Name="jymove.vi" Type="VI" URL="jymove.vi"/>
			<Item Name="ccd.vi" Type="VI" URL="ccd.vi"/>
			<Item Name="Spectral mode.vi" Type="VI" URL="Spectral mode.vi"/>
			<Item Name="Image mode.vi" Type="VI" URL="Image mode.vi"/>
			<Item Name="weitiaoji2.vi" Type="VI" URL="weitiaoji2.vi"/>
		</Item>
		<Item Name="Dependencies" Type="Dependencies"/>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
