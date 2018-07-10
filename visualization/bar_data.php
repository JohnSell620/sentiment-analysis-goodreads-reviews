<?php

header('Access-Control-Allow-Origin: *');

// Database credentials
define('DB_HOST', 'localhost');
define('DB_USER', 'root');
define('DB_PASSWORD', 'root');
define('DB_DATABASE', 'goodreads');

// Data source name
$dsn = 'mysql:host='.DB_HOST.';dbname='.DB_DATABASE.';charset=utf8';

// Database handle
$dbh = new PDO($dsn, DB_USER, DB_PASSWORD);

// Set exception mode for PDO
$dbh->setAttribute(PDO::ATTR_DEFAULT_FETCH_MODE, PDO::FETCH_OBJ);

// Retrieve genres
$sth = $dbh->prepare("SELECT DISTINCT genre FROM reviews");
if ($sth->execute())
{
  $genres = $sth->fetchAll();
}
else
{
  echo "Database genre query unsuccessful.";
}

$obj = array();
foreach ($genres as $genre)
{
  $genre = $genre->genre;
  $genre_obj = new stdClass();
  $genre_obj->genre = $genre;

  $statement = "SELECT reviews.title, reviews.rating, SUM(sentiments.class) as 'sum' FROM sentiments LEFT JOIN reviews ON sentiments.id=reviews.id WHERE reviews.genre=:genre GROUP BY reviews.title, reviews.rating ORDER BY reviews.rating";
  $sth = $dbh->prepare($statement);
  $sth->bindParam(':genre', $genre, PDO::PARAM_STR);

  if ($sth->execute())
  {
    $titles = array();
    $data = new stdClass();
    while ($row = $sth->fetch()) 
    {
      $data->title  = $row->title;

      if ($row->sum < 0)
      {
        $data->value = $row->rating * (1);
        $data->rate  = "Negative";
      }
      else
      {
        $data->value = $row->rating * 1;
        $data->rate  = "Positive";
      }

      array_push($titles, $data);
    }

    $genre_obj->titles = $titles;
    array_push($obj, $genre_obj);
  }
  else
  {
    echo "Database row query unsuccessful.";
  }
}

echo json_encode($obj);
// echo json_last_error();

$sth = null;
$dbh = null;

?>
