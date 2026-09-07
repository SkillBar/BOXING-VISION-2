#ifndef BundleDir
  #error BundleDir must point to the freshly built BoxingVision onedir folder
#endif
#ifndef WebView2Installer
  #error Supply the verified Microsoft Evergreen Standalone x64 installer
#endif

[Setup]
AppId={{BB87A2FC-CF45-4C83-9E5D-4B4C2B9FB531}
AppName=Boxing Vision
AppVersion=0.1.1
DefaultDirName={localappdata}\Programs\BoxingVision
DefaultGroupName=Boxing Vision
UninstallDisplayIcon={app}\BoxingVision.exe
OutputBaseFilename=BoxingVision-Setup-x64
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
MinVersion=10.0.17763
DisableProgramGroupPage=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
; User videos, analyses, model cache and logs in LocalAppData\BoxingVision are
; deliberately not included in the uninstall or install-delete lists.

[Files]
; Extract the offline prerequisite before the large application payload.
Source: "{#WebView2Installer}"; DestName: "WebView2RuntimeInstaller.exe"; Flags: dontcopy
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Boxing Vision"; Filename: "{app}\BoxingVision.exe"
Name: "{autodesktop}\Boxing Vision"; Filename: "{app}\BoxingVision.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"

[Run]
Filename: "{app}\BoxingVision.exe"; Description: "Запустить Boxing Vision"; Flags: nowait postinstall skipifsilent

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Code]
const
  RuntimeKey = 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';

function RuntimeVersionPresent(RootKey: Integer): Boolean;
var
  Version: String;
begin
  Result := RegQueryStringValue(RootKey, RuntimeKey, 'pv', Version);
  if Result then
    Result := (Trim(Version) <> '') and (Version <> '0.0.0.0');
end;

function WebView2Installed: Boolean;
begin
  { Microsoft's documented per-machine 32-bit registry view and per-user key. }
  Result := RuntimeVersionPresent(HKLM32) or RuntimeVersionPresent(HKCU);
end;

function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo,
  MemoTypeInfo, MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := MemoDirInfo + NewLine + NewLine +
    'Python, библиотеки и модели входят в приложение. Настройка вручную не нужна.';
  if not WebView2Installed then
    Result := Result + NewLine + 'Будет установлен Microsoft Edge WebView2 Runtime для окна приложения.';
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ExitCode: Integer;
begin
  Result := '';
  if WebView2Installed then Exit;
  ExtractTemporaryFile('WebView2RuntimeInstaller.exe');
  WizardForm.StatusLabel.Caption := 'Подготовка компонента окна Microsoft WebView2...';
  if not Exec(ExpandConstant('{tmp}\WebView2RuntimeInstaller.exe'), '/silent /install',
    '', SW_HIDE, ewWaitUntilTerminated, ExitCode) then
  begin
    Result := 'Не удалось запустить установку компонента WebView2. Повторите установку Boxing Vision.';
    Exit;
  end;
  Log(Format('WebView2 installer exit code: %d', [ExitCode]));
  if ExitCode = 3010 then
  begin
    NeedsRestart := True;
    Result := 'Для завершения установки WebView2 перезагрузите компьютер и повторно откройте установщик Boxing Vision.';
    Exit;
  end;
  if (ExitCode <> 0) or (not WebView2Installed) then
    Result := 'Компонент WebView2 не установлен. Возможно, установка запрещена политикой компьютера. Обратитесь к отправителю приложения.';
end;
