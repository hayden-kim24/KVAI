-- SQL script generated from CSV

-- Create table to store data fields
CREATE TABLE data_fields (
    id INT AUTO_INCREMENT PRIMARY KEY,
    field_name VARCHAR(255) NOT NULL,
    data_type VARCHAR(50) NOT NULL
);

INSERT INTO data_fields (field_name, data_type) VALUES ("Case Type", "VARCHAR(30)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Filing Number", "VARCHAR(10)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Filing Date", "DATE");
INSERT INTO data_fields (field_name, data_type) VALUES ("Registration Number", "VARCHAR(10)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Registration Date", "VARCHAR(10)");
INSERT INTO data_fields (field_name, data_type) VALUES ("CNR Numbr", "CHAR(16)");
INSERT INTO data_fields (field_name, data_type) VALUES ("First Hearing Date", "DATE");
INSERT INTO data_fields (field_name, data_type) VALUES ("Next Hearing Date", "Date");
INSERT INTO data_fields (field_name, data_type) VALUES ("Decision Date", "Date");
INSERT INTO data_fields (field_name, data_type) VALUES ("Case Stage", "VARCHAR(30)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Nature of Disposal", "VARCHAR(30)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Court Number and Judge", "VARCHAR(30)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Petitioner and Advocate", "VARCHAR(30)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Respondent and Advocate", "VARCHAR(30)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Acts", "Place holder");
INSERT INTO data_fields (field_name, data_type) VALUES ("Under Act(s)", "VARCHAR(30)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Under Section(s)", "Int(10)");
INSERT INTO data_fields (field_name, data_type) VALUES ("FIR Detail", "place holder");
INSERT INTO data_fields (field_name, data_type) VALUES ("Police Station", "VARCHAR(30)");
INSERT INTO data_fields (field_name, data_type) VALUES ("FIR Number", "INT(10)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Year", "INT(4)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Case History", "Place Holder");
INSERT INTO data_fields (field_name, data_type) VALUES ("Judge", "VARCHAR(30)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Business on Date", "DATE");
INSERT INTO data_fields (field_name, data_type) VALUES ("Hearing Date", "DATE");
INSERT INTO data_fields (field_name, data_type) VALUES ("Purpose of Hearing", "VARCHAR(30)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Interm Orders", "Place holder");
INSERT INTO data_fields (field_name, data_type) VALUES ("Order Number", "INT(10)");
INSERT INTO data_fields (field_name, data_type) VALUES ("Order Date", "DATE");
INSERT INTO data_fields (field_name, data_type) VALUES ("Order Details", "VARCHAR(30)");
