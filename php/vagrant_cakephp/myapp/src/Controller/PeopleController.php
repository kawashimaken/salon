<?php
namespace App\Controller;

use App\Controller\AppController;

class PeopleController extends AppController
{

    public function index()
    {
        $data = $this->People->find('all');
        $this->set('data', $data);
    }
    public function add()
    {
        $entity = $this->People->newEntity();
        $this->set('entity', $entity);
    }

    public function create()
    {
        if ($this->request->is('post')) {
            $data = $this->request->data['People'];
            $entity = $this->People->newEntity($data);
            $this->People->save($entity);

        }
        return $this->redirect(['action' => 'index']);
    }

}
