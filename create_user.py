from app import db, User, app

# Fungsi untuk menambahkan pengguna baru
def create_user(username, password):
    try:
        with app.app_context():  # Menambahkan aplikasi konteks Flask
            user = User(username=username, password=password)
            db.session.add(user)
            db.session.commit()
            print(f"User '{username}' created successfully!")
    except Exception as e:
        print(f"Error creating user: {e}")

# Menambahkan beberapa pengguna sebagai contoh
if __name__ == "__main__":
    create_user("admin", "admin123")  # Username dan password untuk admin
    create_user("user1", "password123")  # Username dan password untuk user biasa
