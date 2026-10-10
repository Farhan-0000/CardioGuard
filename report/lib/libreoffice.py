"""
DOCX -> PDF with LibreOffice, refreshing the TOC / List of Figures / List of Tables.

python-docx cannot lay out pages, so the index fields in the DOCX only hold a
placeholder until something updates them. Word does it on open (``updateFields``).
For the PDF, LibreOffice is driven headless with a small Basic macro that loads the
document, updates every document index (twice, so page numbers settle after the
indexes grow), refreshes fields and exports to PDF. The macro lives in a temporary
LibreOffice user profile, so it never touches the user's own profile and does not
clash with a LibreOffice instance that is already running.

If the macro route fails, a plain ``soffice --headless --convert-to pdf`` is used
(indexes then show the placeholder text).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

KNOWN_PATHS: tuple[str, ...] = (
    "/opt/homebrew/bin/soffice",
    "/usr/local/bin/soffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/usr/bin/soffice",
    "/usr/bin/libreoffice",
    "/snap/bin/libreoffice",
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
)

MACRO = """
Sub MedBridgeExport(sIn As String, sOut As String)
    Dim oDoc As Object
    Dim aLoad(1) As New com.sun.star.beans.PropertyValue
    aLoad(0).Name = "Hidden"
    aLoad(0).Value = True
    aLoad(1).Name = "UpdateDocMode"
    aLoad(1).Value = 3
    oDoc = StarDesktop.loadComponentFromURL(ConvertToURL(sIn), "_blank", 0, aLoad())
    Dim i As Integer
    Dim nPass As Integer
    For nPass = 1 To 2
        oDoc.getTextFields().refresh()
        For i = 0 To oDoc.getDocumentIndexes().getCount() - 1
            oDoc.getDocumentIndexes().getByIndex(i).update()
        Next i
    Next nPass
    Dim aPdf(0) As New com.sun.star.beans.PropertyValue
    aPdf(0).Name = "FilterName"
    aPdf(0).Value = "writer_pdf_Export"
    oDoc.storeToURL(ConvertToURL(sOut), aPdf())
    oDoc.close(True)
    StarDesktop.terminate()
End Sub
"""

MODULE_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!DOCTYPE script:module PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" '
    '"module.dtd">\n'
    '<script:module xmlns:script="http://openoffice.org/2000/script" script:name="Module1" '
    'script:language="StarBasic">' + escape(MACRO) + "</script:module>\n"
)


@dataclass
class ConversionResult:
    pdf: Path | None
    method: str  # "macro" | "convert-to" | "unavailable" | "failed"
    detail: str = ""


def find_soffice() -> str | None:
    """LibreOffice executable from PATH or the usual install locations."""
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    for path in KNOWN_PATHS:
        if Path(path).exists():
            return path
    return None


def _profile_url(path: Path) -> str:
    return path.resolve().as_uri()


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False
    )


def _install_macro(soffice: str, profile: Path, timeout: int) -> bool:
    """Initialise a fresh profile and write the export macro into its Standard library."""
    _run(
        [
            soffice,
            f"-env:UserInstallation={_profile_url(profile)}",
            "--headless",
            "--terminate_after_init",
        ],
        timeout,
    )
    std = profile / "user" / "basic" / "Standard"
    if not std.is_dir():
        return False
    (std / "Module1.xba").write_text(MODULE_XML, encoding="utf-8")
    return True


def convert_to_pdf(
    docx: Path, pdf: Path | None = None, *, update_indexes: bool = True, timeout: int = 300
) -> ConversionResult:
    """Convert ``docx`` to PDF next to it (or at ``pdf``); never raises."""
    soffice = find_soffice()
    if soffice is None:
        return ConversionResult(None, "unavailable", "LibreOffice (soffice) not found")
    docx = docx.resolve()
    pdf = (pdf or docx.with_suffix(".pdf")).resolve()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    if pdf.exists():
        pdf.unlink()
    with tempfile.TemporaryDirectory(prefix="medbridge_lo_") as tmp:
        profile = Path(tmp) / "profile"
        # work on a copy with a simple name: macro arguments must not contain quotes/commas
        work_in = Path(tmp) / "report.docx"
        work_out = Path(tmp) / "report.pdf"
        shutil.copyfile(docx, work_in)
        if update_indexes:
            try:
                if _install_macro(soffice, profile, timeout):
                    arg = f'macro:///Standard.Module1.MedBridgeExport("{work_in}","{work_out}")'
                    proc = _run(
                        [
                            soffice,
                            f"-env:UserInstallation={_profile_url(profile)}",
                            "--headless",
                            "--invisible",
                            "--norestore",
                            arg,
                        ],
                        timeout,
                    )
                    if work_out.exists() and work_out.stat().st_size > 0:
                        shutil.move(str(work_out), pdf)
                        return ConversionResult(pdf, "macro", "indexes updated by LibreOffice")
                    detail = (proc.stderr or proc.stdout or "").strip()[-400:]
                else:
                    detail = "could not initialise a LibreOffice profile"
            except (subprocess.TimeoutExpired, OSError) as exc:
                detail = f"macro export failed: {exc}"
        else:
            detail = "index update disabled"
        try:
            outdir = Path(tmp) / "out"
            outdir.mkdir()
            proc = _run(
                [
                    soffice,
                    f"-env:UserInstallation={_profile_url(profile)}",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(outdir),
                    str(work_in),
                ],
                timeout,
            )
            produced = outdir / (work_in.stem + ".pdf")
            if produced.exists():
                shutil.move(str(produced), pdf)
                return ConversionResult(
                    pdf, "convert-to", f"indexes NOT updated ({detail})".strip()
                )
            return ConversionResult(
                None, "failed", (proc.stderr or proc.stdout or "").strip()[-400:]
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return ConversionResult(None, "failed", str(exc))


def pdf_page_count(pdf: Path) -> int | None:
    try:
        import pymupdf  # type: ignore[import-not-found]
    except ImportError:
        try:
            import fitz as pymupdf  # type: ignore[import-not-found,no-redef]
        except ImportError:
            return None
    try:
        with pymupdf.open(str(pdf)) as doc:
            return int(doc.page_count)
    except Exception:
        return None


def platform_hint() -> str:
    system = platform.system()
    if system == "Darwin":
        return "brew install --cask libreoffice"
    if system == "Windows":
        return "winget install TheDocumentFoundation.LibreOffice"
    return "sudo apt install libreoffice-writer"
