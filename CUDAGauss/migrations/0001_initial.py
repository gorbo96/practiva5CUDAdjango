# Generated by Django 3.2.5 on 2021-07-21 18:50

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(default='CUDAGauss/static/images/no-img.jpg', upload_to='CUDAGauss')),
                ('name', models.CharField(max_length=200)),
                ('opcion', models.BooleanField()),
            ],
        ),
    ]
