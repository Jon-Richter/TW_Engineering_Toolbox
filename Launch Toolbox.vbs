Set WshShell = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")
scriptPath = WScript.ScriptFullName
scriptDir = FSO.GetParentFolderName(scriptPath)
WshShell.CurrentDirectory = scriptDir
logDir = scriptDir & "\logs"
If Not FSO.FolderExists(logDir) Then FSO.CreateFolder logDir
logFile = logDir & "\launch_vbs.log"
Function timestamp()
  timestamp = Year(Now) & "-" & Right("0" & Month(Now),2) & "-" & Right("0" & Day(Now),2) & " " & _
              Right("0" & Hour(Now),2) & ":" & Right("0" & Minute(Now),2) & ":" & Right("0" & Second(Now),2)
End Function
Sub Log(msg)
  On Error Resume Next
  Set f = FSO.OpenTextFile(logFile, 8, True)
  f.WriteLine timestamp() & " - " & msg
  f.Close
End Sub
Log "Launch request."
Dim commands()
ReDim commands(7)
commands(0) = Chr(34) & scriptDir & "\.venv\Scripts\pythonw.exe" & Chr(34) & " -m toolbox_app"
commands(1) = Chr(34) & scriptDir & "\.venv\Scripts\python.exe" & Chr(34) & " -m toolbox_app"
commands(2) = "pyw -3 -m toolbox_app"
commands(3) = "py -3 -m toolbox_app"
commands(4) = "pythonw -m toolbox_app"
commands(5) = "python -m toolbox_app"
commands(6) = "python3w -m toolbox_app"
commands(7) = "python3 -m toolbox_app"
launched = False
For i = 0 To UBound(commands)
  cmd = commands(i)
  Log "Trying: " & cmd
  On Error Resume Next
  wnd = 0
  If LCase(Left(Trim(cmd),4)) = "cmd " Then wnd = 1
  ' For GUI apps, use False (don't wait) so the script doesn't hang
  ret = WshShell.Run(cmd, wnd, False)
  errnum = Err.Number
  If errnum <> 0 Then
    Log "Launch failed: " & Err.Description
    Err.Clear
  Else
    Log "Launch succeeded."
    launched = True
    Exit For
  End If
Next
If Not launched Then
  Dim pythonExe
  pythonExe = DetectPythonExecutable()
  If pythonExe <> "" Then
    cmd = Chr(34) & pythonExe & Chr(34) & " -m toolbox_app"
    Log "Trying detected interpreter: " & cmd
    On Error Resume Next
    wnd = 0
    ret = WshShell.Run(cmd, wnd, False)
    errnum = Err.Number
    If errnum <> 0 Then
      Log "Launch failed: " & Err.Description
      Err.Clear
    Else
      Log "Launch succeeded."
      launched = True
    End If
  Else
    Log "No Python interpreter detected for fallback."
  End If
End If
If Not launched Then
  Log "All launch attempts failed."
  WshShell.Popup "Engineering Toolbox failed to start. See: " & logFile, 10, "Launch failed", 16
End If

Function DetectPythonExecutable()
  Dim candidate, resolvedPath
  candidate = WshShell.ExpandEnvironmentStrings("%TOOLBOX_PYTHON_EXE%")
  resolvedPath = ResolvePath(candidate)
  If resolvedPath <> "" Then
    DetectPythonExecutable = resolvedPath
    Exit Function
  End If
  Dim names
  names = Array("pythonw", "pyw", "python3w", "python3", "python", "py")
  For Each candidate In names
    resolvedPath = ResolvePath(candidate)
    If resolvedPath <> "" Then
      DetectPythonExecutable = resolvedPath
      Exit Function
    End If
  Next
  DetectPythonExecutable = ""
End Function

Function ResolvePath(name)
  name = Trim(name)
  If name = "" Then Exit Function
  If InStr(name, "%") > 0 Then name = WshShell.ExpandEnvironmentStrings(name)
  If InStr(name, "\") > 0 Then
    If FSO.FileExists(name) Then
      ResolvePath = name
      Exit Function
    End If
  Else
    Dim candidatePath
    candidatePath = FindCommandPath(name)
    If candidatePath <> "" Then
      ResolvePath = candidatePath
      Exit Function
    End If
  End If
  ResolvePath = ""
End Function

Function FindCommandPath(name)
  Dim exec, output, lines, line
  On Error Resume Next
  Set exec = WshShell.Exec("cmd /c where " & name & " 2>nul")
  Do While exec.Status = 0
    WScript.Sleep 20
  Loop
  output = exec.StdOut.ReadAll
  If Trim(output) = "" Then
    FindCommandPath = ""
    Exit Function
  End If
  lines = Split(output, vbCrLf)
  For Each line In lines
    line = Trim(line)
    If line <> "" And FSO.FileExists(line) Then
      FindCommandPath = line
      Exit Function
    End If
  Next
  FindCommandPath = ""
End Function
