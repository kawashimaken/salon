<html>
<head>
</head>
<body>
<h1>RESULT</h1>
<table>
<?php foreach ($data->toArray() as $obj): ?>
<tr>
<td><?=h($obj->id)?></td>
<td><?=h($obj->name)?></td>
<td><?=h($obj->mail)?></td>
<td><?=h($obj->age)?></td>
</tr>
<?php endforeach;?>
</table>
</body>
</html>