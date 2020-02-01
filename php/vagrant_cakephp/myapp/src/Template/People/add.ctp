
<?=$this->Form->create($entity,
    ['type' => 'post',
        'url' => ['controller' => 'People',
            'action' => 'create']])?>
<fieldset class="form">
	NAME: <?=$this->Form->error('People.name')?>
	<?=$this->Form->text('People.name')?>
	MAIL: <?=$this->Form->error('People.mail')?>
	<?=$this->Form->text('People.mail')?>
	AGE: <?=$this->Form->error('People.age')?>
	<?=$this->Form->text('People.age')?>
	<?=$this->Form->submit('送信')?>
</fieldset>
<?=$this->Form->end()?>

