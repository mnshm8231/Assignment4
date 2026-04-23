import sqlite3
import pdfplumber
import re
import os

SOURCE_DIR = "source"

# (Filename, DB Name, Category, Parser Mode, Use Layout)
PDF_CONFIG = [
    ("ncu1.pdf", "NCU General Regulations", "General", "article", False),
    ("ncu2.pdf", "Course Selection Regulations", "Course", "article", False),
    ("ncu3.pdf", "Credit Transfer Regulations", "Credit", "article", False),
    ("ncu4.pdf", "Grading System Guidelines", "Grade", "article", False),
    ("ncu5.pdf", "Student ID Card Replacement Rules", "Admin", "article", False),
    ("ncu6.pdf", "NCU Student Examination Rules", "Exam", "numbered", True) 
]

def init_db(conn):
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS articles')
    cursor.execute('DROP TABLE IF EXISTS regulations')
    cursor.execute('''CREATE TABLE regulations (reg_id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT)''')
    cursor.execute('''CREATE TABLE articles (art_id INTEGER PRIMARY KEY AUTOINCREMENT, reg_id INTEGER, article_number TEXT, content TEXT, FOREIGN KEY(reg_id) REFERENCES regulations(reg_id))''')
    conn.commit()

def clean_text(text):
    if not text: return ""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_pdf_content(pdf_path, parser_mode="article", use_layout=False):
    print(f"   📂 Reading {os.path.basename(pdf_path)}...")
    articles = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            lines = []
            for page in pdf.pages:
                text = page.extract_text(layout=use_layout)
                if text:
                    lines.extend(text.split('\n'))
            
            current_article_num = None
            current_content = []
            
            if parser_mode == "article":
                pattern = re.compile(r"^\s*Article\s+([0-9]+(?:\-[0-9]+)?)", re.IGNORECASE)
            elif parser_mode == "numbered":
                pattern = re.compile(r"^\s*([0-9]+)\.")

            for line in lines:
                line = line.strip()
                if not line: continue
                if re.match(r"^\d+-\d+$", line) or re.match(r"^Page \d+", line) or (line.isdigit() and len(line) < 4):
                    continue

                match = pattern.match(line)
                
                if match:
                    if current_article_num:
                        full_content = " ".join(current_content)
                        articles.append((current_article_num, clean_text(full_content)))
                    
                    raw_num = match.group(1)
                    if parser_mode == "article":
                        current_article_num = f"Article {raw_num}"
                    else:
                        current_article_num = f"Rule {raw_num}"
                    
                    content_part = line[match.end():].strip()
                    current_content = [content_part] if content_part else []
                else:
                    if current_article_num:
                        current_content.append(line)
            
            if current_article_num:
                full_content = " ".join(current_content)
                articles.append((current_article_num, clean_text(full_content)))
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return []

    return articles

def main():
    db_path = "ncu_regulations.db"
    conn = sqlite3.connect(db_path)
    init_db(conn)
    cursor = conn.cursor()

    print(f"🚀 Starting ETL (Refactored for Clean Data)...")
    reg_id_counter = 1
    total = 0

    for filename, reg_name, category, p_mode, use_layout in PDF_CONFIG:
        file_path = os.path.join(SOURCE_DIR, filename)
        if not os.path.exists(file_path): continue
            
        cursor.execute('INSERT INTO regulations VALUES (?, ?, ?)', (reg_id_counter, reg_name, category))
        extracted = parse_pdf_content(file_path, p_mode, use_layout)
        
        if extracted:
            print(f"      ✅ {filename}: Saved {len(extracted)} articles.")
        else:
            print(f"      ⚠️ WARNING: 0 articles found in {filename}!")

        for art_num, content in extracted:
            cursor.execute('INSERT INTO articles (reg_id, article_number, content) VALUES (?, ?, ?)',
                           (reg_id_counter, art_num, content))
        
        total += len(extracted)
        reg_id_counter += 1

    conn.commit()
    conn.close()
    print("-" * 40)
    print(f"🎉 Database Ready! Total: {total}")

if __name__ == "__main__":
    main()