# app/database_types.py
import os
import json
from cryptography.fernet import Fernet
from sqlalchemy.types import TypeDecorator, LargeBinary

# --- Encryption Setup ---
# 1. Load the secret key from environment variables
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY is not set in the environment.")

# 2. Create the cipher suite
cipher_suite = Fernet(ENCRYPTION_KEY.encode())


class EncryptedJSON(TypeDecorator):
    """
    A custom SQLAlchemy type to transparently encrypt and decrypt JSON data.

    It stores the encrypted data as bytes in the database.
    """
    # This is the database type that will be used to store the encrypted data.
    impl = LargeBinary

    def process_bind_param(self, value, dialect):
        """
        This is called when data is being SENT TO the database.
        It encrypts the value before storing it.
        """
        if value is not None:
            # 1. Convert the Python object (list/dict) to a JSON string
            json_string = json.dumps(value)
            # 2. Encode the string to bytes
            byte_data = json_string.encode('utf-8')
            # 3. Encrypt the bytes
            encrypted_data = cipher_suite.encrypt(byte_data)
            return encrypted_data
        return value

    def process_result_value(self, value, dialect):
        """
        This is called when data is being READ FROM the database.
        It decrypts the value before returning it to the application.
        """
        if value is not None:
            # 1. Decrypt the bytes from the database
            decrypted_data = cipher_suite.decrypt(value)
            # 2. Decode the bytes back to a JSON string
            json_string = decrypted_data.decode('utf-8')
            # 3. Convert the JSON string back to a Python object
            return json.loads(json_string)
        return value