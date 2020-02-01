<?php
namespace App\Controller;

use App\Controller\AppController;

class Hello2Controller extends AppController
{

    public function index()
    {
        $this->viewBuilder()->autoLayout(false);
        $this->set('title', 'This Hello2 Title');
        $this->set('message', 'This is a message');
    }

}
