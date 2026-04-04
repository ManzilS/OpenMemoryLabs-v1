import email
import re
from email.policy import default
from pathlib import Path
import hashlib
from oml.models.schema import Document

def generate_doc_id(content: str) -> str:
    """Generates a stable ID based on content hash."""
    return hashlib.md5(content.encode()).hexdigest()

def clean_email_text(text: str) -> str:
    """
    Cleans email text by removing headers, signatures, and reply blocks.
    Basic heuristics.
    """
    lines = text.splitlines()
    clean_lines = []
    
    # Common separators for replies/forwards
    reply_patterns = [
        r"^-+\s*Original Message\s*-+$",
        r"^On .* wrote:$",
        r"^From:.*Sent:.*To:.*Subject:.*",
        r"^>.*" # Quoted lines
    ]
    
    for line in lines:
        line_stripped = line.strip()
        
        # improved heuristics could go here, for now stop at first clear reply marker
        is_reply = False
        for pattern in reply_patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                is_reply = True
                break
        
        if is_reply:
            break
            
        clean_lines.append(line)
        
    return "\n".join(clean_lines).strip()

def parse_email_file(file_path: Path) -> Document:
    """Parses a single email file into a Document object."""
    try:
        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=default)
    except Exception:
        # Fallback for weird encodings if standard binary read fails
        with open(file_path, "r", encoding="latin-1", errors="replace") as f:
             msg = email.message_from_file(f, policy=default)

    # Extract body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get("Content-Disposition"))
            
            # skip attachments
            if "attachment" in cdispo:
                continue
                
            if ctype == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body += payload.decode("utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode("utf-8", errors="replace")
        else:
            # Fallback if decode=True fails or returns None for str payload
            body = str(msg.get_payload())

    # Metadata
    author = msg.get("From", "Unknown")
    
    # improved recipient handling
    recipients = []
    if msg["To"]:
        recipients.extend([addr.strip() for addr in msg["To"].split(",")])
    if msg["Cc"]:
        recipients.extend([addr.strip() for addr in msg["Cc"].split(",")])
        
    subject = msg.get("Subject", "")
    
    # Timestamp
    date_str = msg.get("Date")
    timestamp = None
    if date_str:
        try:
            timestamp = email.utils.parsedate_to_datetime(date_str)
        except Exception:
            pass

    clean_text = clean_email_text(body)
    
    # Generate ID hash from raw text + subject to be stable
    doc_id = generate_doc_id(body + subject)

    return Document(
        doc_id=doc_id,
        source=str(file_path), # Store full path for reference
        timestamp=timestamp,
        author=author,
        recipients=recipients,
        subject=subject,
        thread_id=None, # Todo: implement threading logic later
        raw_text=body,
        clean_text=clean_text,
        doc_type="email"
    )
