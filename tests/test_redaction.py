from app.privacy.redaction import redact


def test_ssn_masked():
    r = redact("My SSN is 123-45-6789 thanks")
    assert "123-45-6789" not in r.redacted_text
    assert r.counts.get("ssn") == 1


def test_credit_card_luhn_masked():
    r = redact("Card: 4111 1111 1111 1111 on file")
    assert "4111" not in r.redacted_text
    assert r.counts.get("credit_card") == 1


def test_non_card_number_not_card_masked():
    r = redact("Reference 1234 5678 9012 3456")  # fails Luhn
    assert r.counts.get("credit_card") is None


def test_email_and_phone():
    r = redact("Reach me at jordan@example.com or (704) 555-0182.")
    assert "jordan@example.com" not in r.redacted_text
    assert "555-0182" not in r.redacted_text
    assert r.counts.get("email") == 1
    assert r.counts.get("phone") == 1


def test_policy_number():
    r = redact("Policy Number: HX-4482913A applies.")
    assert "HX-4482913A" not in r.redacted_text


def test_address():
    r = redact("Premises at 412 Maple Grove Lane, Apt 3B")
    assert "[ADDRESS REDACTED]" in r.redacted_text


def test_plain_text_untouched():
    text = "The rent is due on the first of the month."
    assert redact(text).redacted_text == text
