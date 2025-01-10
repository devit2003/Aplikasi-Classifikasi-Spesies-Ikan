import os
import nltk
import fitz  # PyMuPDF untuk ekstraksi teks PDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Pastikan untuk mengunduh resource NLTK yang diperlukan
nltk.download('punkt')

# Fungsi untuk membaca teks dari file PDF
def load_documents_from_folder(folder_path):
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):  # Cek file PDF
            doc_text = extract_text_from_pdf(os.path.join(folder_path, filename))
            documents[filename] = doc_text
    return documents

# Fungsi untuk mengekstrak teks dari file PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # Membuka file PDF
    text = ""
    for page in doc:
        text += page.get_text()  # Mengambil teks dari setiap halaman
    return text

# Fungsi untuk membersihkan dan memisahkan teks menjadi kalimat
def clean_and_split_text(text, keyword):
    keyword_tokens = set(keyword.lower().split())  # Tokenisasi kata kunci sebagai set
    sentences = re.split(r'(?<=\.)\s+', text.strip())  # Pisahkan teks menjadi kalimat berdasarkan titik

    relevant_sentences = []
    for sentence in sentences:
        sentence_tokens = set(re.findall(r'\b\w+\b', sentence.lower()))  # Tokenisasi kalimat sebagai set
        if keyword_tokens.issubset(sentence_tokens):  # Cek apakah semua kata kunci ada dalam kalimat
            relevant_sentences.append(sentence.strip())

    return relevant_sentences

# Fungsi untuk menghasilkan deskripsi dari kalimat relevan
def generate_description(relevant_sentences, keyword):
    if not relevant_sentences:
        return "No relevant sentences found."

    # Gabungkan semua kalimat menjadi satu teks tanpa batas panjang
    combined_text = " ".join(relevant_sentences)

    # Buat deskripsi tanpa batas panjang
    description = (
        f"The keyword '{keyword}' is discussed in the following context: "
        f"{combined_text}"
    )
    return description

# Fungsi untuk memproses dokumen dan mencari kata kunci menggunakan VSM
def process_documents_with_vsm(documents, keyword):
    titles = list(documents.keys())
    paragraphs = list(documents.values())

    # Gabungkan kata kunci ke dalam teks untuk analisis VSM
    keyword = keyword.lower()

    # Inisialisasi TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Menghitung TF-IDF untuk dokumen
    tfidf_matrix = vectorizer.fit_transform(paragraphs)

    # Membuat representasi kata kunci menggunakan VSM
    keyword_vector = vectorizer.transform([keyword])

    # Hitung kesamaan kosinus antara kata kunci dan dokumen
    similarities = cosine_similarity(keyword_vector, tfidf_matrix)

    # Menampilkan kalimat relevan yang mengandung kata kunci
    relevant_sentences_by_doc = {}
    for i, similarity in enumerate(similarities[0]):
        if similarity > 0.1:  # Ambil dokumen dengan kesamaan lebih dari 10%
            relevant_sentences_by_doc[titles[i]] = clean_and_split_text(paragraphs[i], keyword)

    return relevant_sentences_by_doc

# Fungsi untuk mencari dokumen yang relevan
def search_documents_for_keyword(keyword, folder_path):
    # Muat dokumen dari folder
    documents = load_documents_from_folder(folder_path)

    # Proses dokumen dan cari dokumen yang relevan dengan kata kunci
    relevant_sentences_by_doc = process_documents_with_vsm(documents, keyword)

    return relevant_sentences_by_doc

# Fungsi utama untuk menampilkan hasil pencarian
def main():
    folder_path = 'doc'  # Ganti dengan path folder yang berisi dokumen .pdf
    keyword = input("Enter the keyword: ")

    # Muat dokumen dari folder dan proses dengan VSM
    relevant_sentences_by_doc = search_documents_for_keyword(keyword, folder_path)

    # Tampilkan hasil pencarian kalimat relevan sebagai deskripsi
    if relevant_sentences_by_doc:
        print("\nGenerated descriptions for the keyword '{}':".format(keyword))
        for title, sentences in relevant_sentences_by_doc.items():
            description = generate_description(sentences, keyword)
            print(f"\nDocument: {title}\n{description}")
    else:
        print("No relevant sentences found.")

if __name__ == '__main__':
    main()
