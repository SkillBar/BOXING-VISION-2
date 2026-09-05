#ifndef BundleDir
  #error BundleDir must point to the freshly built BoxingVision onedir folder
#endif

[Setup]
AppId={{BB87A2FC-CF45-4C83-9E5D-4B4C2B9FB531}
AppName=Boxing Vision
AppVersion=0.1.0
DefaultDirName={localappdata}\Programs\BoxingVision
DefaultGroupName=Boxing Vision
UninstallDisplayIcon={app}\BoxingVision.exe
OutputBaseFilename=BoxingVision-Setup-x64
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
; User videos, analyses, model cache and logs in LocalAppData\BoxingVision are
; deliberately not included in the uninstall or install-delete lists.

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Boxing Vision"; Filename: "{app}\BoxingVision.exe"
Name: "{autodesktop}\Boxing Vision"; Filename: "{app}\BoxingVision.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; Flags: unchecked

[Run]
Filename: "{app}\BoxingVision.exe"; Description: "Запустить Boxing Vision"; Flags: nowait postinstall skipifsilent
