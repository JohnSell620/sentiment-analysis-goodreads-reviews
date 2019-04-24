sqoop import \
--connect jdbc:mysql://localhost/goodreads \
--table reviews \
--username root --password "" \
--taget-dir /goodreads
