from app.parsing.parser import parse_bytes, parse_text


def test_plain_text_roundtrip():
    doc = parse_text("Hello lease world", filename="x.txt")
    assert doc.full_text == "Hello lease world"
    assert doc.pages[0].number == 1


def test_locate_finds_page():
    doc = parse_text("The security deposit is $2,775.00.")
    assert doc.locate("security deposit") == 1
    assert doc.locate("unicorns and rainbows") is None


def test_pdf_extraction():
    import fitz

    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Monthly rent of $1,850.00 is due. Tenant and Landlord agree.")
    data = pdf.tobytes()
    pdf.close()

    doc = parse_bytes(data, "lease.pdf")
    assert doc.source == "pdf"
    assert "$1,850.00" in doc.full_text


def test_bytes_route_text():
    doc = parse_bytes(b"just some plain text", "notes.txt")
    assert doc.full_text == "just some plain text"
