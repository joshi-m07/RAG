import requests

ASK_API = "http://localhost:9000/ask"
INSERT_API = "http://localhost:9000/insert"
UPDATE_API = "http://localhost:9000/update"
RECEIVE_API = "http://localhost:9000/receive"

def ask_query():
    query = input("\n❓ Enter your query: ")
    resp = requests.post(ASK_API, json={"query": query})
    print("💡 Answer:", resp.json()["answer"])

def insert_detail():
    detail = input("\n➕ Enter new detail: ")
    resp = requests.post(INSERT_API, json={"detail": detail})
    print("✅ Insert response:", resp.json())

def update_detail():
    idx = int(input("\n✏️ Enter index to update: "))
    new_detail = input("Enter new detail: ")
    resp = requests.post(UPDATE_API, json={"index": idx, "new_detail": new_detail})
    print("✅ Update response:", resp.json())

def view_details():
    resp = requests.get(RECEIVE_API)
    data = resp.json()
    print("\n📋 Stored details:")
    for i, d in enumerate(data["stored_details"]):
        print(f"  {i}. {d}")

def main():
    while True:
        print("\n--- MENU ---")
        print("1. Ask a query")
        print("2. Insert new detail")
        print("3. Update a detail")
        print("4. View all details")
        print("5. Exit")

        choice = input("Choose option: ")
        if choice == "1":
            ask_query()
        elif choice == "2":
            insert_detail()
        elif choice == "3":
            update_detail()
        elif choice == "4":
            view_details()
        elif choice == "5":
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
