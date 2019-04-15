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

// Statement handle
$statement = <<<HERE
  SELECT sentiments.class, reviews.rating, reviews.genre, reviews.title,
  sentiments.id FROM sentiments LEFT JOIN reviews ON sentiments.id=reviews.id
HERE;
$sth = $dbh->prepare($statement);

if ($sth->execute())
{
  $nodes = array();
  while ($row = $sth->fetch())
  {
	  array_push($nodes, $row);
  }
	$dbh = null;

  // create link object to store reviews-title links
  $link = new stdClass();
  $Links = array();

  // this works since the reviews are stored in order by title in MySQL
  for ($i = 0; $i < sizeof($nodes)-1; $i++)
  {
    $curr_id = $nodes[$i]->id;
    $next_id = $nodes[$i+1]->id;
    $curr_title = $nodes[$i]->title;
    $next_title = $nodes[$i+1]->title;

    if ($curr_title == $next_title)
    {
      $link = new stdClass();
      $link->source = $curr_id;
      $link->target = $next_id;
      $link->title = $curr_title;
      $link->value = $nodes[$i]->rating;
      array_push($Links, $link);
    }
  }

  // combine nodes and Links arrays
  $data = array();
  $data['nodes'] = $nodes;
  $data['links'] = $Links;

  echo json_encode($data);
  // echo json_last_error();
}
else {
  echo "Database query unsuccessful.";
}

// Close database and return null
$sth = null;
$dbh = null;

?>
