from information import CONNECT
import psycopg2
import pickle

#connect db and build the table
if __name__ == '__main__':
    command = """
    CREATE TABLE gdb7244 (
    entry_id serial PRIMARY KEY,
    PBE0_atomization_energies real NOT NULL,
    zindo_excitation_energy_with_the_most_absorption real NOT NULL,
    zindo_highest_absorption_intensity real NOT NULL,
    zindo_homo real NOT NULL,
    zindo_lumo real NOT NULL,
    zindo_1st_excitation_energy real NOT NULL,
    zindo_ionization_potential real NOT NULL,
    zindo_electron_affinity real NOT NULL,
    PBE0_homo real NOT NULL,
    PBE0_lumo real NOT NULL,
    GW_homo real NOT NULL,
    GW_lumo real NOT NULL,
    PBE0_polarizability real NOT NULL,
    SCS_polarizability real NOT NULL,
    coulumb real[23][23] NOT NULL
    );
    """
    db_conn = None
    try:
        db_conn = psycopg2.connect(CONNECT)
        cur = db_conn.cursor()
        cur.execute(command)
        cur.close()
        db_conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if db_conn is not None:
            db_conn.close()
    

    


    
    






