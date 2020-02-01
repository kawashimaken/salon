<?php
namespace App\Controller;

use App\Controller\AppController;

class HelloController extends AppController
{
    public $autoRender = false;

    private $data = [
        ['name' => 'taro', 'mail' => 'taro@gmail.com'],
        ['name' => 'jiro', 'mail' => 'jiro@gmail.com'],
    ];

    public function index()
    {
        if (isset($this->request->query['id'])) {
            $id = $this->request->query['id'];
        }

        echo "<html><head></head><body>";
        echo "<h1>Hello</h1>";
        if (isset($this->request->query['id'])) {
            echo "<h1>ID is " . $id . "</h1>";
            echo json_encode($this->data[$id]);
        }
        echo "</body></html>";
    }

}
