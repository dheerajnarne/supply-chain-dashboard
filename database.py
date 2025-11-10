import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, PrimaryKeyConstraint, ForeignKeyConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import text
import sys

# --- 1. CONFIGURATION ---
# IMPORTANT: Update these values
DATABASE_URL = "postgresql://postgres:%40Dheeraj123@localhost:5432/supply_chain_db"
CSV_FILE_PATH = "DataCoSupplyChainDataset.csv" # Change this to the path of your flat file

# --- 2. DATABASE SCHEMA DEFINITION (SQLAlchemy ORM) ---

# Create a "base" class for our table definitions
Base = declarative_base()

# --- DEFINE PARENT TABLES FIRST ---
# These tables do not depend on any others

class Product(Base):
    __tablename__ = 'products'
    # Column names from your file: "Product Card Id", "Product Name", "Category Name", "Department Name"
    product_card_id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String)
    category_name = Column(String)
    department_name = Column(String)

class Customer(Base):
    __tablename__ = 'customers'
    # Columns: "Customer Id", "Customer Fname", "Customer Lname", "Customer City", 
    #          "Customer State", "Customer Zipcode", "Customer Segment", "Latitude", "Longitude"
    customer_id = Column(Integer, primary_key=True, index=True)
    customer_fname = Column(String)
    customer_lname = Column(String)
    customer_city = Column(String)
    customer_state = Column(String)
    customer_zipcode = Column(Integer)
    customer_segment = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)


# --- DEFINE CHILD TABLES LAST ---
# These tables depend on the parent tables above

class Order(Base):
    __tablename__ = 'orders'
    # Columns: "Order Id", "Customer Id", "Order Date (DateOrders)", "Order Status", 
    #          "Order City", "Order State", "Shipping Mode", "Days for shipping (real)", "Days for shipment (scheduled)"
    
    order_id = Column(Integer, index=True) # Part of composite key
    customer_id = Column(Integer, ForeignKey('customers.customer_id')) # Depends on 'customers'
    order_date = Column(DateTime(timezone=True), index=True) # Part of composite key
    
    order_status = Column(String)
    order_city = Column(String)
    order_state = Column(String)
    shipping_mode = Column(String)
    days_for_shipping_real = Column(Integer)
    days_for_shipment_scheduled = Column(Integer)

    # Define the composite primary key
    __table_args__ = (
        PrimaryKeyConstraint('order_id', 'order_date'),
    )

class OrderItem(Base):
    __tablename__ = 'order_items'
    
    order_item_id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Columns for composite foreign key
    order_id = Column(Integer, index=True) 
    order_date = Column(DateTime(timezone=True))
    
    product_card_id = Column(Integer, ForeignKey('products.product_card_id'), index=True) # Depends on 'products'

    order_item_quantity = Column(Integer)
    order_item_product_price = Column(Float)
    order_item_discount = Column(Float)
    sales = Column(Float)
    order_profit_per_order = Column(Float)
    
    # Define the composite foreign key
    __table_args__ = (
        ForeignKeyConstraint(
            ['order_id', 'order_date'],
            ['orders.order_id', 'orders.order_date'] # Depends on 'orders'
        ),
    )


# --- 3. DATABASE SETUP FUNCTION ---

def setup_database(engine):
    """
    Creates all tables and converts 'orders' to a TimescaleDB hypertable.
    """
    print("Creating tables in the database...")
    try:
        # Create all tables defined in Base
        # SQLAlchemy is smart enough to create them in the correct dependency order
        Base.metadata.create_all(engine)
        print("Tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")
        sys.exit(1)

    # Convert 'orders' table to a hypertable
    print("Converting 'orders' table to a hypertable...")
    with engine.connect() as conn:
        try:
            # We use text() to execute raw SQL
            conn.execute(text("SELECT create_hypertable('orders', 'order_date');"))
            conn.commit() # Must commit after DDL
            print("Hypertable created successfully.")
        except OperationalError as e:
            if "already a hypertable" in str(e):
                print("Table 'orders' is already a hypertable.")
            else:
                print(f"Error creating hypertable: {e}")
                conn.rollback()
                sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during hypertable creation: {e}")
            conn.rollback()
            sys.exit(1)

# --- 4. DATA INGESTION FUNCTION ---

def ingest_data(engine, file_path):
    """
    Reads the CSV, normalizes the data, and inserts it into the database tables.
    """
    try:
        print(f"Loading data from '{file_path}'...")
        df = pd.read_csv(file_path, encoding='latin1') # 'latin1' is common for mixed-char datasets
        print(f"Loaded {len(df)} rows from CSV.")
        
        # Strip whitespace from all column names
        df.columns = df.columns.str.strip()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
        
    # --- Data Cleaning ---
    try:
        # Use the correct lowercase column name
        print("Starting 'order date (DateOrders)' conversion...") 
        
        df['order date (DateOrders)'] = pd.to_datetime(
            df['order date (DateOrders)'], 
            errors='coerce'
        )
        
        failed_rows = df['order date (DateOrders)'].isnull().sum()
        if failed_rows > 0:
            print(f"Warning: {failed_rows} rows had unparseable dates and will be dropped.")
            df.dropna(subset=['order date (DateOrders)'], inplace=True)

        # Use the correct lowercase column name in the log
        print(f"Successfully converted 'order date (DateOrders)' to datetime. {len(df)} valid rows remaining.")

    except KeyError as e:
        print(f"\n--- FATAL ERROR: Column Not Found ---")
        print(f"Error: The column {e} was not found in your CSV file.")
        print("This is likely because the column name in the CSV is misspelled or has spaces.")
        print("\nHere is a list of the ACTUAL column names found in the file:")
        print(df.columns.to_list())
        print("\nPlease update the `database.py` script to match the correct column name.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during date conversion: {e}")
        sys.exit(1)

    # --- Ingest Products ---
    print("Processing Products...")
    df_products = df[[
        'Product Card Id', 'Product Name', 'Category Name', 'Department Name'
    ]].drop_duplicates(subset=['Product Card Id']).rename(columns={
        'Product Card Id': 'product_card_id',
        'Product Name': 'product_name',
        'Category Name': 'category_name',
        'Department Name': 'department_name'
    })
    # Use a session to handle potential integrity errors gracefully
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        df_products.to_sql('products', session.bind, if_exists='append', index=False)
        session.commit()
    except Exception as e:
        print(f"Error ingesting products (likely duplicates): {e}")
        session.rollback()
    finally:
        session.close()
    print(f"Ingested {len(df_products)} unique products.")

    # --- Ingest Customers ---
    print("Processing Customers...")
    df_customers = df[[
        'Customer Id', 'Customer Fname', 'Customer Lname', 'Customer City', 
        'Customer State', 'Customer Zipcode', 'Customer Segment', 'Latitude', 'Longitude'
    ]].drop_duplicates(subset=['Customer Id']).rename(columns={
        'Customer Id': 'customer_id',
        'Customer Fname': 'customer_fname',
        'Customer Lname': 'customer_lname',
        'Customer City': 'customer_city',
        'Customer State': 'customer_state',
        'Customer Zipcode': 'customer_zipcode',
        'Customer Segment': 'customer_segment',
        'Latitude': 'latitude',
        'Longitude': 'longitude'
    })
    session = Session()
    try:
        df_customers.to_sql('customers', session.bind, if_exists='append', index=False)
        session.commit()
    except Exception as e:
        print(f"Error ingesting customers (likely duplicates): {e}")
        session.rollback()
    finally:
        session.close()
    print(f"Ingested {len(df_customers)} unique customers.")

    # --- Ingest Orders ---
    print("Processing Orders...")
    
    # FIX 1: Use the correct lowercase 'order date (DateOrders)'
    df_orders = df[[
        'Order Id', 'Customer Id', 'order date (DateOrders)', 'Order Status', 
        'Order City', 'Order State', 'Shipping Mode', 'Days for shipping (real)', 
        'Days for shipment (scheduled)'
    ]].drop_duplicates(subset=['Order Id', 'order date (DateOrders)']).rename(columns={
        'Order Id': 'order_id',
        'Customer Id': 'customer_id',
        'order date (DateOrders)': 'order_date', # FIX 2: Use lowercase key
        'Order Status': 'order_status',
        'Order City': 'order_city',
        'Order State': 'order_state',
        'Shipping Mode': 'shipping_mode',
        'Days for shipping (real)': 'days_for_shipping_real',
        'Days for shipment (scheduled)': 'days_for_shipment_scheduled'
    })
    session = Session()
    try:
        df_orders.to_sql('orders', session.bind, if_exists='append', index=False)
        session.commit()
    except Exception as e:
        print(f"Error ingesting orders (likely duplicates): {e}")
        session.rollback()
    finally:
        session.close()
    print(f"Ingested {len(df_orders)} unique orders.")

    # --- Ingest Order Items ---
    print("Processing Order Items...")
    
    # FIX 3: Use the correct lowercase 'order date (DateOrders)'
    df_items = df[[
        'Order Id', 'Product Card Id', 'order date (DateOrders)', 
        'Order Item Quantity', 'Order Item Product Price', 
        'Order Item Discount', 'Sales', 'Order Profit Per Order'
    ]].rename(columns={
        'Order Id': 'order_id',
        'Product Card Id': 'product_card_id',
        'order date (DateOrders)': 'order_date', # FIX 4: Use lowercase key
        'Order Item Quantity': 'order_item_quantity',
        'Order Item Product Price': 'order_item_product_price',
        'Order Item Discount': 'order_item_discount',
        'Sales': 'sales',
        'Order Profit Per Order': 'order_profit_per_order'
    })
    # We add an index to serve as the 'order_item_id' primary key
    df_items.reset_index(inplace=True)
    df_items.rename(columns={'index': 'order_item_id'}, inplace=True)
    
    session = Session()
    try:
        df_items.to_sql('order_items', session.bind, if_exists='append', index=False)
        session.commit()
    except Exception as e:
        print(f"Error ingesting order items: {e}")
        session.rollback()
    finally:
        session.close()
    print(f"Ingested {len(df_items)} order items.")
    
    print("\n--- Data ingestion complete! ---")

# --- 5. EXECUTION ---

if __name__ == "__main__":
    try:
        engine = create_engine(DATABASE_URL)
        print(f"Connecting to database at {DATABASE_URL}...")
        engine.connect()
        print("Connection successful.")
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        print("Please check your DATABASE_URL, database status, and 'psycopg2-binary' installation.")
        sys.exit(1)

    # Step 1: Create the tables and hypertable
    setup_database(engine)
    
    # Step 2: Load and ingest the data from the CSV
    # We wrap this in a session to handle the whole ingest as a transaction
    print("\n--- Starting Data Ingestion ---")
    
    # A bit redundant since to_sql uses its own, but good practice
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        ingest_data(engine, CSV_FILE_PATH)
        session.commit()
    except Exception as e:
        print(f"A fatal error occurred during ingestion: {e}")
        session.rollback()
    finally:
        session.close()