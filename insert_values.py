from information import CONNECT
import psycopg2
import pickle
import numpy as np


if __name__ == "__main__":
    coulombs = pickle.load(open("coulombs.pickle",'rb'))
    properties = pickle.load(open("properties.pickle",'rb'))
    db_conn = None
    
    sqlcmd = """INSERT INTO gdb7244 (entry_id,
                                    PBE0_atomization_energies,
                                    zindo_excitation_energy_with_the_most_absorption,
                                    zindo_highest_absorption_intensity,
                                    zindo_homo,
                                    zindo_lumo,
                                    zindo_1st_excitation_energy,
                                    zindo_ionization_potential,
                                    zindo_electron_affinity,
                                    PBE0_homo,
                                    PBE0_lumo,
                                    GW_homo,
                                    GW_lumo,
                                    PBE0_polarizability,
                                    SCS_polarizability,
                                    coulumb
                                    ) 
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
    
    try:
        db_conn = psycopg2.connect(CONNECT)
        cur = db_conn.cursor()
        for i, (cou, pro) in enumerate(zip(coulombs, properties)):
            cou_str = str(map(str, cou))
            cur.execute(sqlcmd,(i, pro[0], pro[1], pro[2], pro[3], pro[4], \
                                pro[5], pro[6], pro[7], pro[8], pro[9], pro[10],\
                                 pro[11], pro[12], pro[13], cou))
        cur.close()
        db_conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if db_conn is not None:
            db_conn.close()
    
    
    
        
    
    




