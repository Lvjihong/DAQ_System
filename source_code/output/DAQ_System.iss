; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "3Vision_DAQ_System"
#define MyAppVersion "1.0"
#define MyAppPublisher "LvJihong"
#define MyAppExeName "DAQ_System.exe"
#define MyAppAssocName MyAppName + " File"
#define MyAppAssocExt ".myp"
#define MyAppAssocKey StringChange(MyAppAssocName, " ", "") + MyAppAssocExt

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{6494A3DA-AE03-4E79-8276-DDB42F847EC7}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
ChangesAssociations=yes
DisableProgramGroupPage=yes
; Uncomment the following line to run in non administrative install mode (install for current user only.)
;PrivilegesRequired=lowest
OutputBaseFilename=3VisionDAQsetup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "F:\MicrosoftVisualStudio\Source\Repos\DAQ_System\x64\Release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "F:\MicrosoftVisualStudio\Source\Repos\DAQ_System\x64\Release\*.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "F:\Qt\Qt5.14.2\5.14.2\msvc2017_64\bin\Qt5Core.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "F:\Qt\Qt5.14.2\5.14.2\msvc2017_64\bin\Qt5Gui.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "F:\Qt\Qt5.14.2\5.14.2\msvc2017_64\bin\Qt5Widgets.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "F:\Qt\Qt5.14.2\5.14.2\msvc2017_64\plugins\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "C:\Windows\System32\kernel32.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Windows\SysWOW64\shell32.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Windows\SysWOW64\advapi32.dll"; DestDir: "{app}"; Flags: ignoreversion
Source: "F:\MicrosoftVisualStudio\Source\Repos\DAQ_System\weights\last_best.onnx"; DestDir: "{app}\weights"; Flags: ignoreversion
Source: "F:\MicrosoftVisualStudio\Source\Repos\DAQ_System\config\config_cpp.xml"; DestDir: "{app}\config"; Flags: ignoreversion
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Dirs]
; 在安装目录下创建文件夹
Name: "F:\DAQ_System\data\show_data"
Name: "F:\DAQ_System\data\saved_data"

[Registry]
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocExt}\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppAssocKey}"; ValueData: ""; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}"; ValueType: string; ValueName: ""; ValueData: "{#MyAppAssocName}"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKA; Subkey: "Software\Classes\{#MyAppAssocKey}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""
Root: HKA; Subkey: "Software\Classes\Applications\{#MyAppExeName}\SupportedTypes"; ValueType: string; ValueName: ".myp"; ValueData: ""

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

